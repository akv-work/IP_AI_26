import io
import random
from pathlib import Path

import time, shutil, subprocess, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ROOT
from google.colab import files
from ultralytics.trackers.basetrack import BaseTrack
from IPython.display import HTML, display
from base64 import b64encode

random.seed(42)
print("GPU доступен:", torch.cuda.is_available())


WEIGHTS = Path("runs/detect/runs_plates/yolo12n/weights/best.pt")

if not WEIGHTS.exists():
    print("best.pt не найден — выбери файл с компьютера")
    up = files.upload()
    WEIGHTS = Path(list(up.keys())[0])

print("Веса:", WEIGHTS)


up = files.upload()
Path(list(up)[0]).rename("video_full.mp4")

START, DURATION = 0, 20

cap = cv2.VideoCapture("video.mp4")
print("кадров:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "| fps:", cap.get(cv2.CAP_PROP_FPS),
      "| размер:", int(cap.get(3)), "x", int(cap.get(4)))
cap.release()


TRACKERS_DIR = ROOT / "cfg" / "trackers"
Path("trackers").mkdir(exist_ok=True)

for name in ["botsort.yaml", "bytetrack.yaml"]:
    shutil.copy(TRACKERS_DIR / name, Path("trackers") / name)

print(" bytetrack.yaml ")
print(Path("trackers/bytetrack.yaml").read_text())
print(" botsort.yaml ")
print(Path("trackers/botsort.yaml").read_text())


VIDEO = "video.mp4"
CONF = 0.25

def run_tracking(tracker_cfg, label, folder, conf=CONF, save=True):
    BaseTrack.reset_id()               
    model = YOLO(str(WEIGHTS))         
    tracks = {}                        
    n_dets, n_frames = 0, 0

    t0 = time.time()
    for i, r in enumerate(model.track(
            source=VIDEO, tracker=tracker_cfg, conf=conf, imgsz=640,
            stream=True, save=save, project="runs_track", name=folder,
            exist_ok=True, verbose=False)):
        n_frames += 1
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            n_dets += len(ids)
            for tid in ids:
                tracks.setdefault(tid, []).append(i)
    dt = time.time() - t0

    lens = np.array([len(v) for v in tracks.values()]) if tracks else np.array([0])
    stats = {
        "эксперимент": label,
        "уникальных ID": len(tracks),
        "средняя длина трека (кадров)": round(float(lens.mean()), 1),
        "коротких треков (<15 кадров)": int((lens < 15).sum()),
        "объектов на кадр": round(n_dets / max(n_frames, 1), 2),
        "FPS": round(n_frames / dt, 1),
    }
    return stats, tracks


RESULTS, ALL_TRACKS = [], {}

BASE = [
    ("BoT-SORT (по умолчанию)",  "botsort_default",   "trackers/botsort.yaml"),
    ("ByteTrack (по умолчанию)", "bytetrack_default", "trackers/bytetrack.yaml"),
]

for label, folder, cfg in BASE:
    stats, tracks = run_tracking(cfg, label, folder, save=True)
    RESULTS.append(stats)
    ALL_TRACKS[label] = tracks
    print("готово:", label)

pd.DataFrame(RESULTS)


def make_cfg(base_yaml, out_name, **changes):
    cfg = yaml.safe_load(Path(base_yaml).read_text())
    cfg.update(changes)
    out = Path("trackers") / out_name
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return str(out)

BYTE, BOT = "trackers/bytetrack.yaml", "trackers/botsort.yaml"

EXPERIMENTS = [
    ("ByteTrack: track_buffer=10", "byte_buf10", BYTE, dict(track_buffer=10)),
    ("ByteTrack: track_buffer=90", "byte_buf90", BYTE, dict(track_buffer=90)),
    ("ByteTrack: track_high_thresh=0.5", "byte_high05", BYTE, dict(track_high_thresh=0.5)),
    ("ByteTrack: match_thresh=0.5", "byte_match05", BYTE, dict(match_thresh=0.5)),
    ("ByteTrack: match_thresh=0.95", "byte_match095",BYTE, dict(match_thresh=0.95)),
    ("ByteTrack: conf детектора=0.1", "byte_conf01",  BYTE, dict(conf=0.1)),
    ("BoT-SORT: track_buffer=10", "bot_buf10", BOT, dict(track_buffer=10)),
    ("BoT-SORT: track_buffer=90", "bot_buf90", BOT, dict(track_buffer=90)),
    ("BoT-SORT: gmc_method=none", "bot_nogmc", BOT, dict(gmc_method="none")),
    ("BoT-SORT: conf детектора=0.1", "bot_conf01", BOT, dict(conf=0.1)),
    ("BoT-SORT: with_reid=True", "bot_reid", BOT, dict(with_reid=True)),
]

for label, folder, base, changes in EXPERIMENTS:
    changes = dict(changes)
    conf = changes.pop("conf", CONF)          
    cfg_path = make_cfg(base, f"{folder}.yaml", **changes)
    try:
        stats, tracks = run_tracking(cfg_path, label, folder, conf=conf, save=False)
    except Exception as e:
        print(f"{label}: пропущено ({type(e).__name__}: {e})")
        continue
    RESULTS.append(stats)
    ALL_TRACKS[label] = tracks
    print("готово:", label)


df = pd.DataFrame(RESULTS)
cols = ["уникальных ID", "средняя длина трека (кадров)", "коротких треков (<15 кадров)"]

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
for ax, col in zip(axes, cols):
    ax.barh(df["эксперимент"], df[col])
    ax.set_title(col)
    ax.grid(alpha=0.3, axis="x")
axes[0].invert_yaxis()
plt.tight_layout()
plt.show()

df


labels = ["BoT-SORT (по умолчанию)", "ByteTrack (по умолчанию)"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
for ax, label in zip(axes, labels):
    items = sorted(ALL_TRACKS[label].items(), key=lambda kv: kv[1][0])
    for row, (tid, frames) in enumerate(items):
        ax.hlines(row, frames[0], frames[-1], linewidth=3)
    ax.set_title(f"{label}\nтреков: {len(items)}")
    ax.set_xlabel("номер кадра")
    ax.set_ylabel("трек (по порядку появления)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


def show_tracked(folder):
    root = Path("/content/runs")
    found = list(root.rglob(f"{folder}/*.avi")) + list(root.rglob(f"{folder}/*.mp4"))
    if not found:
        print("видео не найдено для:", folder)
        return
    out = f"{folder}_h264.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(found[0]),
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", out])
    data = b64encode(open(out, "rb").read()).decode()
    display(HTML(f'<video height="480" controls><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))

show_tracked("bytetrack_default")