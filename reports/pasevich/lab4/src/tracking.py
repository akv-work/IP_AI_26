import os
import cv2
from ultralytics import YOLO

WEIGHTS = "runs/detect/vehicles_yolo12s/weights/best.pt"
VIDEO_PATH = "videos/traffic1.mp4"
OUTPUT_DIR = "tracking_results"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_tracking(model, tracker_config, run_name):
    print(f"\n{'=' * 60}")
    print(f"ЗАПУСК: {tracker_config} -> {run_name}")
    print(f"{'=' * 60}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(OUTPUT_DIR, f"{run_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    active_ids = set()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        annotated = results[0].plot()
        writer.write(annotated)
        frame_count += 1

        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for tid in track_ids:
                active_ids.add(tid)

        if frame_count % 30 == 0:
            print(f"  Кадров: {frame_count}/{total_frames}")

    cap.release()
    writer.release()

    print(f"\n  Готово: {out_path}")
    print(f"  Кадров: {frame_count}")
    print(f"  Уникальных ID: {len(active_ids)}")

    return {
        "run": run_name,
        "tracker": tracker_config,
        "frames": frame_count,
        "unique_ids": len(active_ids),
        "output": out_path,
    }


if __name__ == "__main__":
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"Детектор не найден: {WEIGHTS}")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

    model = YOLO(WEIGHTS)

    experiments = [
        ("bytetrack.yaml", "01_bytetrack_default"),
        ("botsort.yaml", "02_botsort_default"),
        ("bytetrack_custom.yaml", "03_bytetrack_custom"),
        ("botsort_custom.yaml", "04_botsort_custom"),
    ]

    all_results = []

    for tracker, name in experiments:
        try:
            result = run_tracking(model, tracker, name)
            all_results.append(result)
        except Exception as e:
            print(f"  Ошибка с {tracker}: {e}")

    print(f"\n{'=' * 60}")
    print("ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'=' * 60}")
    print(f"{'Запуск':<30} {'Кадров':<10} {'Уникальных ID':<15}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['run']:<30} {r['frames']:<10} {r['unique_ids']:<15}")