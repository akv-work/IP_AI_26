import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
WEIGHTS = str(BASE / "best.pt")
OUT_DIR = BASE / "results"
CFG_DIR = BASE


def resolve_video(value):
    if value:
        p = Path(value)
        if p.exists():
            return str(p)
        if (BASE / value).exists():
            return str(BASE / value)
        raise SystemExit(f"Видео не найдено: {value}")
    found = sorted(BASE.glob("*.mp4"))
    if not found:
        raise SystemExit("Положите .mp4 рядом с lab4.py или укажите --video")
    print("Видео:", found[0].name)
    return str(found[0])

BYTETRACK = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}
BOTSORT = {
    **BYTETRACK,
    "tracker_type": "botsort",
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.8,
    "with_reid": False,
}

EXPERIMENTS = {
    "bytetrack_default": (BYTETRACK, {}),
    "bytetrack_buffer10": (BYTETRACK, {"track_buffer": 10}),
    "bytetrack_buffer90": (BYTETRACK, {"track_buffer": 90}),
    "bytetrack_high0.5": (BYTETRACK, {"track_high_thresh": 0.5}),
    "bytetrack_low0.3": (BYTETRACK, {"track_low_thresh": 0.3}),
    "bytetrack_new0.5": (BYTETRACK, {"new_track_thresh": 0.5}),
    "bytetrack_match0.6": (BYTETRACK, {"match_thresh": 0.6}),
    "bytetrack_match0.95": (BYTETRACK, {"match_thresh": 0.95}),
    "botsort_default": (BOTSORT, {}),
    "botsort_buffer10": (BOTSORT, {"track_buffer": 10}),
    "botsort_buffer90": (BOTSORT, {"track_buffer": 90}),
    "botsort_high0.5": (BOTSORT, {"track_high_thresh": 0.5}),
    "botsort_match0.6": (BOTSORT, {"match_thresh": 0.6}),
    "botsort_match0.95": (BOTSORT, {"match_thresh": 0.95}),
    "botsort_gmc_none": (BOTSORT, {"gmc_method": "none"}),
    "botsort_prox0.3": (BOTSORT, {"proximity_thresh": 0.3}),
    "botsort_reid": (BOTSORT, {"with_reid": True}),
}


def write_cfg(name, base, changes):
    CFG_DIR.mkdir(exist_ok=True)
    cfg = {**base, **changes}
    path = CFG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def download(args):
    import yt_dlp
    out = BASE / args.out
    opts = {"outtmpl": str(out), "format": "mp4[height<=720]/best[height<=720]",
            "merge_output_format": "mp4"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([args.url])
    print("Сохранено:", out)


def run_tracking(video, cfg_path, name, weights, conf, imgsz, save_video=True, device=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    frames = 0
    dets_total = 0
    track_len = defaultdict(int)
    first_frame, last_frame = {}, {}
    per_frame_ids = []
    t0 = time.time()

    total = 0
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception:
        pass

    results = model.track(source=video, tracker=str(cfg_path), conf=conf, imgsz=imgsz,
                          stream=True, persist=True, save=save_video,
                          project=str(OUT_DIR), name=name, exist_ok=True,
                          device=device, verbose=False)
    for r in results:
        frames += 1
        ids = []
        if r.boxes is not None and r.boxes.id is not None:
            ids = [int(i) for i in r.boxes.id.tolist()]
        dets_total += len(r.boxes) if r.boxes is not None else 0
        for i in ids:
            track_len[i] += 1
            first_frame.setdefault(i, frames)
            last_frame[i] = frames
        per_frame_ids.append(len(ids))
        if frames % 20 == 0:
            done = f"{frames}/{total}" if total > 0 else str(frames)
            pct = f" ({100 * frames / total:.0f}%)" if total > 0 else ""
            print(f"\r  [{name}] кадр {done}{pct}, {frames / (time.time() - t0):.1f} FPS",
                  end="", flush=True)
    print()
    elapsed = time.time() - t0

    n_tracks = len(track_len)
    lens = list(track_len.values())
    gaps = sum((last_frame[i] - first_frame[i] + 1) - track_len[i] for i in track_len)
    short = sum(1 for l in lens if l < 10)
    return {
        "run": name,
        "frames": frames,
        "unique_ids": n_tracks,
        "mean_track_len": round(sum(lens) / n_tracks, 1) if n_tracks else 0,
        "max_track_len": max(lens) if lens else 0,
        "short_tracks(<10)": short,
        "gap_frames": gaps,
        "avg_objects_per_frame": round(sum(per_frame_ids) / frames, 2) if frames else 0,
        "fps": round(frames / elapsed, 1) if elapsed else 0,
    }


def track(args):
    video = resolve_video(args.video)
    if args.cfg:
        cfg_path = Path(args.cfg)
        if not cfg_path.exists():
            cfg_path = BASE / args.cfg
    else:
        base = BYTETRACK if args.tracker == "bytetrack" else BOTSORT
        cfg_path = write_cfg(f"{args.tracker}_default", base, {})
    name = f"{Path(video).stem}_{cfg_path.stem}"
    res = run_tracking(video, cfg_path, name, args.weights, args.conf, args.imgsz,
                       device=args.device)
    for k, v in res.items():
        print(f"{k:>24}: {v}")
    print("Видео с треками:", OUT_DIR / name)


def experiments(args):
    video = resolve_video(args.video)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (base, changes) in EXPERIMENTS.items():
        cfg_path = write_cfg(name, base, changes)
        run_name = f"{Path(video).stem}_{name}"
        print(f"\n=== {name}  {changes or '(по умолчанию)'} ===")
        try:
            res = run_tracking(video, cfg_path, run_name, args.weights, args.conf,
                               args.imgsz, save_video=args.save, device=args.device)
        except Exception as e:
            print("  пропущено:", e)
            continue
        res["changed_params"] = str(changes) if changes else "default"
        rows.append(res)
        print("  ", {k: res[k] for k in ("unique_ids", "mean_track_len", "short_tracks(<10)", "gap_frames", "fps")})

    csv_path = OUT_DIR / f"{Path(video).stem}_experiments.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nТаблица результатов:", csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--url", required=True)
    d.add_argument("--out", default="video.mp4")

    for cmd in ("track", "experiments"):
        s = sub.add_parser(cmd)
        s.add_argument("--video", default=None, help="по умолчанию — первый .mp4 рядом со скриптом")
        s.add_argument("--weights", default=WEIGHTS)
        s.add_argument("--conf", type=float, default=0.25)
        s.add_argument("--imgsz", type=int, default=640)
        s.add_argument("--device", default=None)
        if cmd == "track":
            s.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
            s.add_argument("--cfg", default=None, help="свой yaml-конфиг трекера")
        else:
            s.add_argument("--save", action="store_true", help="сохранять видео для каждого запуска")

    a = p.parse_args()
    {"download": download, "track": track, "experiments": experiments}[a.cmd](a)