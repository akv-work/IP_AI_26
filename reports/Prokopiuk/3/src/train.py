from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO("yolo12m.yaml")
    
    res = model.train(
        data=BASE_DIR / "rock-paper-scissors-14" / "data.yaml",
        epochs=10,
        imgsz=512,
        project=BASE_DIR,
        name="rsp",
        device=0,
        workers=8,
        exist_ok=True,
        batch=20
    )
    