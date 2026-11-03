from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(BASE_DIR / "rsp12m_clean" / "weights" / "best.pt")

    model.predict(
        source=BASE_DIR / "test",
        imgsz=640,
        conf=0.25,
        save=True,
        project=BASE_DIR,
        name="predictions",
        exist_ok=True,
    )
