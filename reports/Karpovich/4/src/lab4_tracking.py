import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT

EXPERIMENTS = {
    "bytetrack_default": ("bytetrack", {}),
    "bytetrack_high_thresh": ("bytetrack", {"track_high_thresh": 0.5, "new_track_thresh": 0.5}),
    "bytetrack_low_thresh": ("bytetrack", {"track_low_thresh": 0.3}),
    "bytetrack_buffer_10": ("bytetrack", {"track_buffer": 10}),
    "bytetrack_buffer_90": ("bytetrack", {"track_buffer": 90}),
    "bytetrack_match_0.6": ("bytetrack", {"match_thresh": 0.6}),
    "bytetrack_match_0.95": ("bytetrack", {"match_thresh": 0.95}),
    "botsort_default": ("botsort", {}),
    "botsort_high_thresh": ("botsort", {"track_high_thresh": 0.5, "new_track_thresh": 0.5}),
    "botsort_low_thresh": ("botsort", {"track_low_thresh": 0.3}),
    "botsort_buffer_10": ("botsort", {"track_buffer": 10}),
    "botsort_buffer_90": ("botsort", {"track_buffer": 90}),
    "botsort_match_0.6": ("botsort", {"match_thresh": 0.6}),
    "botsort_match_0.95": ("botsort", {"match_thresh": 0.95}),
    "botsort_gmc_none": ("botsort", {"gmc_method": "none"}),
    "botsort_gmc_orb": ("botsort", {"gmc_method": "orb"}),
    "botsort_reid": ("botsort", {"with_reid": True}),
}


def color_for(track_id):
    return (int(track_id) * 37 % 200 + 55, int(track_id) * 91 % 200 + 55, int(track_id) * 53 % 200 + 55)


def build_config(name, tracker_type, overrides, config_dir):
    with open(ROOT / "cfg" / "trackers" / f"{tracker_type}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    path = config_dir / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def video_info(source):
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, total


def run_experiment(weights, source, cfg_path, video_path, args):
    model = YOLO(weights)
    fps, width, height, total = video_info(source)
    stride = max(args.vid_stride, 1)
    total = total // stride
    if args.max_frames:
        total = min(total, args.max_frames)
    writer = None
    if video_path is not None:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height))

    lifetimes = defaultdict(int)
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    active_per_frame = []
    frames = 0
    start = time.time()

    results = model.track(
        source=str(source),
        tracker=str(cfg_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        vid_stride=stride,
        stream=True,
        persist=False,
        verbose=False,
    )

    for result in results:
        frames += 1
        boxes = result.boxes
        ids = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                lifetimes[track_id] += 1
                trails[track_id].append((int((x1 + x2) / 2), int(y2)))
        active_per_frame.append(len(ids))

        if writer is not None:
            frame = result.plot()
            for track_id in ids:
                points = list(trails[track_id])
                for a, b in zip(points[:-1], points[1:]):
                    cv2.line(frame, a, b, color_for(track_id), 2)
            writer.write(frame)

        if frames % args.progress == 0:
            speed = frames / max(time.time() - start, 1e-9)
            left = (total - frames) / max(speed, 1e-9)
            print(f"    кадр {frames}/{total}  {speed:.1f} кадр/с  осталось ~{left:.0f} с", flush=True)

        if args.max_frames and frames >= args.max_frames:
            break

    elapsed = max(time.time() - start, 1e-9)
    if writer is not None:
        writer.release()

    lengths = sorted(lifetimes.values())
    unique_ids = len(lengths)
    max_simultaneous = max(active_per_frame) if active_per_frame else 0
    median_len = lengths[len(lengths) // 2] if lengths else 0
    return {
        "frames": frames,
        "unique_ids": unique_ids,
        "max_simultaneous": max_simultaneous,
        "excess_ids": max(unique_ids - max_simultaneous, 0),
        "mean_active": round(sum(active_per_frame) / max(frames, 1), 2),
        "mean_track_len": round(sum(lengths) / max(unique_ids, 1), 1),
        "median_track_len": median_len,
        "short_tracks": sum(1 for l in lengths if l < args.short_len),
        "fps": round(frames / elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out", default="tracking_results")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trail", type=int, default=40)
    parser.add_argument("--short-len", type=int, default=15)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    config_dir = out_dir / "configs"
    video_dir = out_dir / "videos"
    config_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    names = args.only if args.only else list(EXPERIMENTS)
    rows = []

    for source in args.sources:
        source_name = Path(source).stem
        for name in names:
            tracker_type, overrides = EXPERIMENTS[name]
            cfg_path = build_config(name, tracker_type, overrides, config_dir)
            print(f"[СТАРТ] {source_name} / {name}", flush=True)
            video_path = None if args.no_video else video_dir / f"{source_name}__{name}.mp4"
            try:
                stats = run_experiment(args.weights, source, cfg_path, video_path, args)
            except Exception as error:
                print(f"[ПРОПУСК] {source_name} / {name}: {error}")
                continue
            row = {
                "video": source_name,
                "experiment": name,
                "tracker": tracker_type,
                "params": json.dumps(overrides, ensure_ascii=False),
                **stats,
            }
            rows.append(row)
            print(
                f"{source_name:20s} {name:24s} ids={stats['unique_ids']:4d} "
                f"max_sim={stats['max_simultaneous']:3d} excess={stats['excess_ids']:4d} "
                f"mean_len={stats['mean_track_len']:7.1f} short={stats['short_tracks']:4d} "
                f"fps={stats['fps']:6.1f}"
            )

    if rows:
        csv_path = out_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print("Таблица сохранена:", csv_path)


if __name__ == "__main__":
    main()
