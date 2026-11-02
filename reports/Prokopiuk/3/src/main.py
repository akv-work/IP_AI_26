from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(
        BASE_DIR / "rsp_finetune" / "weights" / "best.pt"
    )

    image_dir = BASE_DIR / "test"

    model.predict(
        source=image_dir,
        imgsz=512,
        conf=0.25,
        save=True,
        project=BASE_DIR,
        name="predictions",
        exist_ok=True
    )

