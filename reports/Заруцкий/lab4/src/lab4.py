import os
import time
import cv2
import yaml
from ultralytics import YOLO

VIDEOS = {
    "one_cat":  "video_one_cat.mp4",
    "two_cats": "video_two_cats.mp4",
}

OUTPUT_DIR = "output"
CONFIG_DIR = "tracker_configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

BYTETRACK_TEMPLATE = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}

BOTSORT_TEMPLATE = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": False,
    "model": "auto",
}


def make_config(tracker: str, params: dict) -> str:
    base = BYTETRACK_TEMPLATE.copy() if tracker == "bytetrack" else BOTSORT_TEMPLATE.copy()
    base.update(params)
    name = f"{tracker}_buf{params.get('track_buffer', 30)}_thr{params.get('track_high_thresh', 0.25)}.yaml"
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(base, f, allow_unicode=True)
    return path


def run_tracking(video_path: str, tracker: str, params: dict, iou: float):
    config_path = make_config(tracker, params)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Не удалось открыть видео: {video_path}")
        return set(), 0.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    w_out, h_out = w // 2, h // 2
    out_name = f"FINAL_{tracker}_{os.path.basename(video_path).replace('.mp4', '')}_iou{iou}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w_out, h_out))

    unique_ids = set()
    t_start = time.perf_counter()

    results = model.track(
        source=video_path,
        tracker=config_path,
        conf=0.35,
        iou=iou,
        imgsz=1280,
        persist=True,
        verbose=False,
        stream=True,
    )

    for r in results:
        frame = r.orig_img.copy()

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
            confs = r.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if tracker == "bytetrack" else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{int(track_id) if track_id else 0}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if track_id is not None:
                    unique_ids.add(int(track_id))

        frame_resized = cv2.resize(frame, (w_out, h_out))
        out.write(frame_resized)

    out.release()
    elapsed = time.perf_counter() - t_start

    print(f"  {tracker:10s} | buffer={params.get('track_buffer', 30):<3} "
          f"| thr={params.get('track_high_thresh', 0.25):<4} "
          f"| match={params.get('match_thresh', 0.8):<4} "
          f"| iou={iou:<4} | IDs={len(unique_ids):<3} | {elapsed:.2f} с")

    return unique_ids, elapsed


def experiment_track_buffer(video_path: str):
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 1: влияние track_buffer")
    print("=" * 70)

    for tracker in ("bytetrack", "botsort"):
        for buf in (5, 60):
            params = {"track_buffer": buf}
            run_tracking(video_path, tracker, params, iou=0.6)


def experiment_thresholds(video_path: str):
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 2: влияние track_high_thresh и match_thresh")
    print("=" * 70)

    for tracker in ("bytetrack", "botsort"):
        for thr in (0.25, 0.5):
            for match in (0.75, 0.8):
                params = {"track_high_thresh": thr, "match_thresh": match}
                run_tracking(video_path, tracker, params, iou=0.6)


def experiment_iou(video_path: str):
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 3: влияние iou детектора")
    print("=" * 70)

    for tracker in ("bytetrack", "botsort"):
        for iou in (0.5, 0.6):
            params = {"track_buffer": 30}
            run_tracking(video_path, tracker, params, iou=iou)


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# ВИДЕО 1: один кот (стабильность ID)")
    print("#" * 70)

    if os.path.exists(VIDEOS["one_cat"]):
        print("\n[Базовый прогон — сравнение скорости]")
        run_tracking(VIDEOS["one_cat"], "bytetrack", {"track_buffer": 30}, iou=0.6)
        run_tracking(VIDEOS["one_cat"], "botsort",   {"track_buffer": 30}, iou=0.6)
    else:
        print(f"[!] Файл не найден: {VIDEOS['one_cat']} — пропускаем.")

    print("\n" + "#" * 70)
    print("# ВИДЕО 2: два кота (проблема перекрытия)")
    print("#" * 70)

    if os.path.exists(VIDEOS["two_cats"]):
        experiment_iou(VIDEOS["two_cats"])
        experiment_track_buffer(VIDEOS["two_cats"])
        experiment_thresholds(VIDEOS["two_cats"])
    else:
        print(f"[!] Файл не найден: {VIDEOS['two_cats']} — пропускаем.")

    print("\nГотово. Все видео сохранены в папку:", OUTPUT_DIR)