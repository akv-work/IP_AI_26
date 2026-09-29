import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = "VbmkPXg"
EPOCHS = 25
IMG_SIZE = 640
BATCH = 16
DEVICE = 0
RUN_NAME = "rps_yolo11m"
INTERNET_IMAGES_DIR = "internet_images"


def main():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
    dataset = project.version(14).download("yolov11")

    print("Датасет скачан в:", dataset.location)

    with open(os.path.join(dataset.location, "data.yaml"), encoding="utf-8") as f:
        original = yaml.safe_load(f)

    fixed = {
        "path": os.path.abspath(dataset.location),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": original["nc"],
        "names": original["names"],
    }
    DATA_YAML = os.path.join(dataset.location, "data_fixed.yaml")
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(fixed, f, allow_unicode=True)

    print("Классы:", fixed["names"])



    model = YOLO("yolo11m.pt")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project="/content/drive/MyDrive/lab3_runs",
        name=RUN_NAME,
        patience=15,
        plots=True,
    )


    BEST_WEIGHTS = os.path.join("/content/drive/MyDrive/lab3_runs", RUN_NAME, "weights", "best.pt")



    best_model = YOLO(BEST_WEIGHTS)
    metrics = best_model.val(data=DATA_YAML, split="test", device=DEVICE)

    print("\n===== РЕЗУЛЬТАТЫ НА ТЕСТЕ =====")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    for i, name in fixed["names"].items() if isinstance(fixed["names"], dict) else enumerate(fixed["names"]):
        print(f"  AP@0.5 для класса '{name}': {metrics.box.ap50[i]:.4f}")


    if os.path.isdir(INTERNET_IMAGES_DIR) and os.listdir(INTERNET_IMAGES_DIR):
        best_model.predict(
            source=INTERNET_IMAGES_DIR,
            conf=0.4,
            save=True,
            project="/content/drive/MyDrive/lab3_runs",
            name="internet_predictions",
            device=DEVICE,
        )
        print("Готово! Смотри результат в папке runs/internet_predictions")
    else:
        print(f"Папка '{INTERNET_IMAGES_DIR}' пуста или не найдена - добавь туда фото и запусти шаг 5 снова.")


if __name__ == "__main__":
    main()
