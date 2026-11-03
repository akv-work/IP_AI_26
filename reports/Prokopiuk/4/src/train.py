from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    # ВАЖНО: .pt (COCO-предобученные веса), а не .yaml (случайная инициализация)
    model = YOLO("yolo12m.pt")

    model.train(
        data=BASE_DIR / "rock-paper-scissors-14" / "data.yaml",
        imgsz=640,          # изображения датасета уже 640x640
        epochs=80,
        batch=16,
        patience=20,
        mosaic=0.6,
        close_mosaic=10,
        scale=0.5,
        hsv_v=0.5,
        hsv_s=0.6,
        fliplr=0.5,
        mixup=0.0,
        copy_paste=0.0,
        project=BASE_DIR,
        name="rsp12m_clean",
        device=0,
        workers=8,
        amp=True,
        exist_ok=True,
    )
