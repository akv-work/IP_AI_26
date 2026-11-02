from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    model = YOLO(BASE_DIR / "rsp_finetune" / "weights" / "best.pt")

    model.train(
        data=BASE_DIR / "rock-paper-scissors-14" / "data.yaml",
        epochs=20,
        imgsz=512,
        batch=24,
        project=BASE_DIR,
        name="rsp_finetune",
        device=0,
        workers=8,
        amp=True,
        exist_ok=True
    )

