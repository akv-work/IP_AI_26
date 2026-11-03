import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "lapx", "roboflow",
"pyyaml"])
import os, time, shutil, collections, getpass
from pathlib import Path
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO
from google.colab import files
DEVICE = "0" if torch.cuda.is_available() else "cpu"
IMGSZ = 640
DET_CONF = 0.10
DET_IOU = 0.5
MAX_FRAMES = 600
OUT_DIR = Path("lab4_out"); OUT_DIR.mkdir(exist_ok=True)
VIDEO_DIR = Path("videos"); VIDEO_DIR.mkdir(exist_ok=True)
CFG_DIR = Path("tracker_cfgs"); CFG_DIR.mkdir(exist_ok=True)
cands = [
 "runs/yolo11m_watermeter/weights/best.pt",
 "runs/detect/runs/yolo11m_watermeter/weights/best.pt",
 "/content/runs/detect/runs/yolo11m_watermeter/weights/best.pt",
 "/content/drive/MyDrive/lab_weights/best.pt",
 "best.pt",
]
WEIGHTS = next((p for p in cands if Path(p).exists()), None)
if WEIGHTS is None:
 from roboflow import Roboflow
 API_KEY = getpass.getpass("Roboflow API key: ").strip()
 EPOCHS, BATCH, RUN_NAME = 20, 16, "yolo11m_watermeter"
 rf = Roboflow(api_key=API_KEY)
 ds = rf.workspace("koer3741-gmail-com").project("watermeteramrv2").version(1).download(
 "yolov8", location="datasets/watermeteramrv2")
 ds_dir = Path(ds.location).resolve()
 dcfg = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8"))
 dcfg["path"] = str(ds_dir)
 for split in ("train", "val", "test"):
 if (ds_dir / split / "images").exists():
 dcfg[split] = f"{split}/images"
 else:
 dcfg.pop(split, None)
 if "val" not in dcfg and "test" in dcfg:
 dcfg["val"] = dcfg["test"]
 data_yaml = ds_dir / "data_fixed.yaml"
 data_yaml.write_text(yaml.safe_dump(dcfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
 m = YOLO("yolo11m.pt")
 m.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
 project="runs", name=RUN_NAME, patience=30, seed=42, exist_ok=True,
 plots=True, verbose=True)
 WEIGHTS = str(Path(m.trainer.best))
print("Weights:", WEIGHTS)
print("Classes:", YOLO(WEIGHTS).names)
VIDEOS = {p.stem: p for p in VIDEO_DIR.glob("*.mp4")}
if not VIDEOS:
 up = files.upload()
 for fn in up:
 dst = VIDEO_DIR / Path(fn).name
 shutil.move(fn, dst)
 VIDEOS[dst.stem] = dst
print("Videos:", {k: str(v) for k, v in VIDEOS.items()})
TR_DIR = Path(ultralytics.__file__).parent / "cfg" / "trackers"
BASE = {t: yaml.safe_load((TR_DIR / f"{t}.yaml").read_text()) for t in ("bytetrack", "botsort")}
def make_cfg(name, tracker, overrides):
 cfg = dict(BASE[tracker])
 cfg.update(overrides)
 path = CFG_DIR / f"{name}.yaml"
 path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
 return path
EXPERIMENTS = [
 ("byte_default", "bytetrack", {}),
 ("byte_buffer10", "bytetrack", {"track_buffer": 10}),
 ("byte_buffer90", "bytetrack", {"track_buffer": 90}),
 ("byte_match0.6", "bytetrack", {"match_thresh": 0.6}),
 ("byte_match0.95", "bytetrack", {"match_thresh": 0.95}),
 ("byte_high0.5", "bytetrack", {"track_high_thresh": 0.5, "new_track_thresh": 0.5}),
 ("byte_low0.3", "bytetrack", {"track_low_thresh": 0.3}),
 ("byte_newtrack0.6", "bytetrack", {"new_track_thresh": 0.6}),
 ("bot_default", "botsort", {}),
 ("bot_gmc_none", "botsort", {"gmc_method": "none"}),
 ("bot_buffer10", "botsort", {"track_buffer": 10}),
 ("bot_buffer90", "botsort", {"track_buffer": 90}),
 ("bot_match0.6", "botsort", {"match_thresh": 0.6}),
 ("bot_match0.95", "botsort", {"match_thresh": 0.95}),
 ("bot_proximity0.3", "botsort", {"proximity_thresh": 0.3}),
 ("bot_proximity0.7", "botsort", {"proximity_thresh": 0.7}),
 ("bot_reid", "botsort", {"with_reid": True}),
]
CFG_PATHS = {n: make_cfg(n, t, o) for n, t, o in EXPERIMENTS}
def run_tracking(video_path, cfg_path):
 model = YOLO(WEIGHTS)
 track_len = collections.Counter()
 per_frame = []
 n_frames = 0
 t0 = time.time()
 gen = model.track(source=str(video_path), tracker=str(cfg_path), stream=True, persist=False,
 conf=DET_CONF, iou=DET_IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)
 for r in gen:
 n_frames += 1
 ids = []
 if r.boxes is not None and r.boxes.id is not None:
 ids = r.boxes.id.int().cpu().tolist()
 for tid in ids:
 track_len[tid] += 1
 per_frame.append(len(ids))
 if MAX_FRAMES and n_frames >= MAX_FRAMES:
 break
 dt = time.time() - t0
 lengths = np.array(list(track_len.values())) if track_len else np.array([0])
 max_simul = max(per_frame) if per_frame else 0
 return {
 "frames": n_frames,
 "unique_ids": len(track_len),
 "max_simultaneous": max_simul,
 "id_excess": len(track_len) - max_simul,
 "mean_objs_per_frame": round(float(np.mean(per_frame)) if per_frame else 0, 2),
 "mean_track_len": round(float(lengths.mean()), 1),
 "median_track_len": float(np.median(lengths)),
 "short_tracks": int((lengths < 15).sum()),
 "fps": round(n_frames / dt, 1),
 }
rows = []
for vname, vpath in VIDEOS.items():
 for exp_name, tracker, overrides in EXPERIMENTS:
 try:
 res = run_tracking(vpath, CFG_PATHS[exp_name])
 except Exception as e:
 print("skip", vname, exp_name, e)
 continue
 rows.append({"video": vname, "experiment": exp_name, "tracker": tracker,
 "params": str(overrides) if overrides else "default", **res})
 print(vname, exp_name, res["unique_ids"], res["id_excess"], res["mean_track_len"], res["short_tracks"],
res["fps"])
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "results.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df)
for vname in df["video"].unique():
 d = df[df["video"] == vname]
 fig, ax = plt.subplots(1, 3, figsize=(18, 5))
 cols = ["tab:blue" if t == "bytetrack" else "tab:orange" for t in d["tracker"]]
 for a, c, title in zip(ax, ["unique_ids", "id_excess", "mean_track_len"],
 ["Unique IDs", "ID excess", "Mean track length"]):
 a.barh(d["experiment"], d[c], color=cols)
 a.set_title(f"{vname}: {title}")
 a.invert_yaxis()
 a.grid(alpha=0.3)
 plt.tight_layout()
 plt.savefig(OUT_DIR / f"metrics_{vname}.png", dpi=130)
 plt.show()
src = str(next(iter(VIDEOS.values())))
cap = cv2.VideoCapture(src)
fps = cap.get(5) or 25
W, H = int(cap.get(3)), int(cap.get(4))
cap.release()
def render(name, tracker=None, max_frames=MAX_FRAMES):
 model = YOLO(WEIGHTS)
 raw = str(OUT_DIR / f"{name}_raw.mp4")
 out = str(OUT_DIR / f"{name}.mp4")
 w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
 if tracker:
 gen = model.track(source=src, tracker=tracker, stream=True, persist=False,
 conf=0.25, imgsz=IMGSZ, device=DEVICE, verbose=False)
 else:
 gen = model.predict(source=src, stream=True, conf=0.25, imgsz=IMGSZ, device=DEVICE, verbose=False)
 for i, r in enumerate(gen):
 w.write(r.plot())
 if i + 1 >= max_frames:
 break
 w.release()
 subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
 "-vcodec", "libx264", "-pix_fmt", "yuv420p", out])
 os.remove(raw)
 return out
render("1_detection")
render("2_bytetrack", "bytetrack.yaml")
render("3_botsort", "botsort.yaml")
shutil.copytree(CFG_DIR, OUT_DIR / "tracker_cfgs", dirs_exist_ok=True)
shutil.make_archive("lab4_results", "zip", OUT_DIR)
files.download("lab4_results.zip")
