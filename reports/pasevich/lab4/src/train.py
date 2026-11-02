import os
import yaml
from ultralytics import YOLO

DATASET_YAML = "./vehicles-q0x2v/data.yaml"
MODEL_NAME = "yolo12s.pt"
PROJECT_DIR = "runs/detect"
RUN_NAME = "vehicles_yolo12s"

EPOCHS = 5
IMGSZ = 416
BATCH = 4
DEVICE = "cpu"
WORKERS = 2


def train():
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer='SGD',
        lr0=0.01,
        patience=10,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        cache=False,
    )


def validate():
    best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(best):
        return
    model = YOLO(best)
    metrics = model.val(
        data=DATASET_YAML,
        split='test',
        imgsz=IMGSZ,
        conf=0.25,
        iou=0.7,
        device=DEVICE,
    )
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    train()
    validate()