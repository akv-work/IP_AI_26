!pip install -q ultralytics roboflow
from pathlib import Path
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
 
API_KEY = "<API_KEY>"
root = Path("/content/datasets/watermeteramrv2-v1")
if not root.exists():
    Roboflow(api_key=API_KEY).workspace("koer3741-gmail-com").project("watermeteramrv2").version(1).download("yolov8", location=str(root))
 
cfg = yaml.safe_load((root / "data.yaml").read_text())
cfg.update(path=str(root), train="train/images", val="valid/images", test="test/images")
(root / "data.yaml").write_text(yaml.safe_dump(cfg))
 
model = YOLO("yolo12m.pt")
model.train(data=str(root / "data.yaml"), epochs=30, imgsz=640, batch=16, device=0,
            project="/content/runs", name="yolo12m_watermeter", patience=30, seed=42, exist_ok=True)
 
best = YOLO(str(Path(model.trainer.save_dir) / "weights" / "best.pt"))
m = best.val(data=str(root / "data.yaml"), split="test", project="/content/runs", name="test", exist_ok=True)
print("mAP50:", m.box.map50, "mAP50-95:", m.box.map, "P:", m.box.mp, "R:", m.box.mr)
