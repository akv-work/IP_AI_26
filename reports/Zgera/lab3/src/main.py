import os
import sys
import subprocess
from pathlib import Path
API_KEY = " 7Je6nZGW89BZcZV0Bbfx"
EPOCHS = 20
IMGSZ = 640
BATCH = 16
PHOTO = "1.png"
CONF = 0.25
RUN_NAME = "yolo11m_watermeter"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "roboflow", "pyyaml",
"opencv-python-headless", "matplotlib"])
import yaml
import cv2
import torch
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO
device = "0" if torch.cuda.is_available() else "cpu"
rf = Roboflow(api_key=API_KEY.strip())
ds = rf.workspace("koer3741-gmail-com").project("watermeteramrv2").version(1).download("yolov8",
location="datasets/watermeteramrv2")
ds_dir = Path(ds.location).resolve()
cfg = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8"))
cfg["path"] = str(ds_dir)
for split in ("train", "val", "test"):
 if (ds_dir / split / "images").exists():
 cfg[split] = f"{split}/images"
 else:
 cfg.pop(split, None)
if "val" not in cfg and "test" in cfg:
 cfg["val"] = cfg["test"]
data_yaml = ds_dir / "data_fixed.yaml"
data_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
model = YOLO("yolo11m.pt")
model.train(
 data=str(data_yaml),
 epochs=EPOCHS,
 imgsz=IMGSZ,
 batch=BATCH,
 device=device,
 project="runs",
 name=RUN_NAME,
 patience=30,
 seed=42,
 exist_ok=True,
 plots=True,
 verbose=True
)
best = Path("runs") / RUN_NAME / "weights" / "best.pt"
model = YOLO(str(best))
split = "test" if "test" in cfg else "val"
m = model.val(data=str(data_yaml), split=split, imgsz=IMGSZ, batch=BATCH, device=device, plots=True,
verbose=True)
print(f"Precision: {m.box.mp:.4f}")
print(f"Recall: {m.box.mr:.4f}")
print(f"mAP@0.5: {m.box.map50:.4f}")
print(f"mAP@0.5:0.95: {m.box.map:.4f}")
if not Path(PHOTO).exists():
 raise FileNotFoundError(PHOTO)
res = model.predict(source=PHOTO, conf=CONF, imgsz=IMGSZ, device=device, verbose=False)[0]
for b in res.boxes:
 x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
 cls_id = int(b.cls.item())
 score = float(b.conf.item())
 print(f"{model.names[cls_id]} {score:.2f} [{x1}, {y1}, {x2}, {y2}]")
img_bgr = res.plot()
cv2.imwrite("result_1.png", img_bgr)
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
