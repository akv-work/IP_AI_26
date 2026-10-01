import os
import yaml
from ultralytics import YOLO

DATASET_YAML = "./vehicles-q0x2v/data.yaml"
MODEL_NAME = "yolo12s.pt"
PROJECT_DIR = "runs/detect"
RUN_NAME = "vehicles_yolo12s"

EPOCHS = 8
IMGSZ = 416
BATCH = 4
DEVICE = "cpu"
WORKERS = 2


def check_data_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден.\n"
            f"Сначала запустите download_dataset.py"
        )

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    print("=" * 60)
    print("Содержимое data.yaml:")
    for key in ["path", "train", "val", "test", "nc", "names"]:
        if key in data:
            print(f"  {key}: {data[key]}")
    print("=" * 60)

    return data


def train():
    check_data_yaml(DATASET_YAML)

    print(f"\nЗагрузка модели: {MODEL_NAME}")
    print("(при первом запуске веса скачаются автоматически, ~19 МБ)")
    model = YOLO(MODEL_NAME)

    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ НА CPU")
    print(f"  Эпох:       {EPOCHS}")
    print(f"  Размер:     {IMGSZ}")
    print(f"  Batch:      {BATCH}")
    print(f"  Устройство: {DEVICE}")
    print("ВНИМАНИЕ: обучение на CPU может занять несколько часов!")
    print("=" * 60 + "\n")

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

    print(f"\nОбучение завершено: {PROJECT_DIR}/{RUN_NAME}")


def validate():
    best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

    if not os.path.exists(best):
        print(f"best.pt не найден: {best}")
        return

    print(f"\nВалидация на тестовой выборке: {best}")
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
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
    validate()