import os
import glob
import cv2
import yaml
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO


VIDEO_PATH = "test_video.mp4"
WEIGHTS_PATH = None  # None = автопоиск best.pt из ЛР3
OUTPUT_DIR = "lab4_output"
IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = 0  # 0 для GPU, 'cpu' для процессора


SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"



def download_video_if_needed():
    """Скачивает тестовое видео с GitHub, если его нет локально"""
    if os.path.exists(VIDEO_PATH):
        print(f"[INFO] Видео {VIDEO_PATH} уже существует.")
        return True

    print(f"[INFO] Видео не найдено. Скачиваем тестовое видео с GitHub...")
    try:

        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, VIDEO_PATH)
        print(f"[INFO] Видео успешно скачано и сохранено как {VIDEO_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Не удалось скачать видео: {e}")
        print("Пожалуйста, скачайте любое видео с машинами вручную и назовите его test_video.mp4")
        return False



def create_tracker_configs():
    botsort_cfg = {
        'tracker_type': 'botsort',
        'track_high_thresh': 0.3,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.4,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'fuse_score': True,
        'gmc_method': 'sparseOptFlow',
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'with_reid': False
    }

    bytetrack_cfg = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.4,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'fuse_score': True
    }

    with open("botsort_custom.yaml", "w") as f:
        yaml.dump(botsort_cfg, f, default_flow_style=False)
    with open("bytetrack_custom.yaml", "w") as f:
        yaml.dump(bytetrack_cfg, f, default_flow_style=False)
    print("[INFO] Конфигурационные файлы трекеров созданы.")


def find_best_weights():
    if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    candidates = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
    if not candidates:
        raise FileNotFoundError("Не найдены веса best.pt. Укажите путь в WEIGHTS_PATH.")
    return max(candidates, key=os.path.getmtime)



def process_video(video_path, tracker_yaml, model, output_dir, tracker_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{video_name}_{tracker_name}.mp4")
    csv_path = os.path.join(output_dir, f"{video_name}_{tracker_name}_stats.csv")


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    track_history = defaultdict(lambda: {"frames": 0})
    frame_stats = []
    frame_idx = 0

    print(f"\n[INFO] Запуск трекера: {tracker_name} ({tracker_yaml})")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame, persist=True, tracker=tracker_yaml,
            imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, verbose=False
        )[0]

        boxes = results.boxes
        annotated = frame.copy()
        num_active = 0

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.int().cpu().tolist()
            num_active = len(track_ids)

            for tid, cls_id, (x1, y1, x2, y2) in zip(track_ids, clss, xyxy):
                track_history[tid]["frames"] += 1

                label = f"ID:{tid} {model.names[cls_id]}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(y1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.putText(annotated, f"Tracker: {tracker_name} | Frame: {frame_idx}/{total_frames} | Active: {num_active}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(annotated)
        frame_stats.append({"frame": frame_idx, "active_tracks": num_active})

    cap.release()
    out.release()

    df = pd.DataFrame(frame_stats)
    df.to_csv(csv_path, index=False)

    total_unique = len(track_history)
    active_tracks = [t for t, d in track_history.items() if d["frames"] > 1]
    short_tracks = total_unique - len(active_tracks)
    avg_lifetime = np.mean([d["frames"] for d in track_history.values()]) if track_history else 0.0

    summary = {
        "Tracker": tracker_name,
        "Unique_IDs": total_unique,
        "Active_IDs": len(active_tracks),
        "Short_IDs": short_tracks,
        "Avg_Lifetime_Frames": round(float(avg_lifetime), 2)
    }
    print(f"  -> Уникальных ID: {total_unique} | Стабильных: {len(active_tracks)} | Коротких: {short_tracks}")
    return summary, df



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    if not download_video_if_needed():
        return


    create_tracker_configs()
    weights = find_best_weights()
    model = YOLO(weights)
    print(f"[INFO] Модель: {weights}")


    botsort_summary, df_botsort = process_video(
        VIDEO_PATH, "botsort_custom.yaml", model, OUTPUT_DIR, "BoT-SORT"
    )


    bytetrack_summary, df_bytetrack = process_video(
        VIDEO_PATH, "bytetrack_custom.yaml", model, OUTPUT_DIR, "ByteTrack"
    )


    df_summary = pd.DataFrame([botsort_summary, bytetrack_summary])
    summary_path = os.path.join(OUTPUT_DIR, "trackers_summary.csv")
    df_summary.to_csv(summary_path, index=False)


    plt.figure(figsize=(12, 5))
    plt.plot(df_botsort["frame"], df_botsort["active_tracks"], label="BoT-SORT", color="#2E86AB", linewidth=1.5)
    plt.plot(df_bytetrack["frame"], df_bytetrack["active_tracks"], label="ByteTrack", color="#C73E1D", linewidth=1.5)
    plt.xlabel("Номер кадра", fontsize=12)
    plt.ylabel("Количество активных объектов", fontsize=12)
    plt.title("Сравнение количества одновременно отслеживаемых объектов", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "trackers_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 50)
    print(" РАБОТА УСПЕШНО ЗАВЕРШЕНА!")
    print(f" Все результаты сохранены в папку: {OUTPUT_DIR}/")
    print(f" Видео BoT-SORT: {OUTPUT_DIR}/{VIDEO_PATH.replace('.mp4', '')}_BoT-SORT.mp4")
    print(f"Видео ByteTrack: {OUTPUT_DIR}/{VIDEO_PATH.replace('.mp4', '')}_ByteTrack.mp4")
    print(f" Сводная таблица: {summary_path}")
    print(f" График сравнения: {plot_path}")
    print("\n[Сводная таблица]:")
    print(df_summary.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    main()