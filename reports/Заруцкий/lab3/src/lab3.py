import os
import cv2
import yaml
import shutil
import urllib.request
import torch
from roboflow import Roboflow
from ultralytics import YOLO

API_KEY = "ВАШ_API_KEY_ROBOFLOW"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("mohamed-traore-2ekkp").project("cats-n9b87")
dataset = project.version(3).download("yolov11")

print("Датасет скачан в:", dataset.location)

yaml_path = os.path.join(dataset.location, "data.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

data["train"] = os.path.join(dataset.location, "train", "images")
data["val"] = os.path.join(dataset.location, "valid", "images")
data["test"] = os.path.join(dataset.location, "test", "images")

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True)

print("data.yaml:")
print(open(yaml_path, encoding="utf-8").read())

model = YOLO("yolo11s.pt")

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/detect",
    name="cats_yolo11s",
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    patience=100,
    val=True,
    plots=True,
    save=True,
    verbose=True,
    device=0 if torch.cuda.is_available() else "cpu",
)

MODEL_PATH = os.path.join("runs", "detect", "cats_yolo11s", "weights", "best.pt")
print("Лучшие веса:", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("Валидация модели на тестовой выборке...")
metrics = model.val(data=yaml_path, split="test")

print("\nРезультаты:")
print(f"  mAP50    : {metrics.box.map50:.4f}")
print(f"  mAP50-95 : {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall   : {metrics.box.mr:.4f}")

os.makedirs("web_cats", exist_ok=True)

urls = [
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640",
    "https://images.unsplash.com/photo-1519052537078-e6302a4968d4?w=640",
    "https://images.unsplash.com/photo-1495360010541-f48722b34f7d?w=640",
]

for i, url in enumerate(urls, 1):
    try:
        urllib.request.urlretrieve(url, f"web_cats/cat{i}.jpg")
        print(f"Скачано: web_cats/cat{i}.jpg")
    except Exception as e:
        print(f"Ошибка загрузки {url}: {e}")

os.makedirs("results", exist_ok=True)

images = sorted(
    f for f in os.listdir("web_cats")
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

for img_name in images:
    input_path = os.path.join("web_cats", img_name)
    output_path = os.path.join("results", img_name)

    print(f"\nОбработка: {img_name}...")

    results = model.predict(
        source=input_path,
        conf=0.25,
        save=False,
        verbose=False,
    )

    annotated_frame = results[0].plot()
    cv2.imwrite(output_path, annotated_frame)
    print(f"Сохранено: {output_path}, найдено котов: {len(results[0].boxes)}")

print("\nВсе картинки сохранены в папку 'results'")

try:
    from google.colab import files
    from IPython.display import Image, display

    print("Выберите картинку с котом на вашем компьютере:")
    uploaded = files.upload()

    for filename in uploaded.keys():
        print(f"\nОбработка файла: {filename}")

        results = model.predict(source=filename, conf=0.25, save=False)
        annotated_frame = results[0].plot()

        output_path = os.path.join("results", filename)
        cv2.imwrite(output_path, annotated_frame)

        print(f"Найдено котов: {len(results[0].boxes)}")
        display(Image(filename=output_path))

    shutil.make_archive("results", "zip", "results")
    files.download("results.zip")
    files.download(MODEL_PATH)

except ImportError:
    print("Загрузка через files.upload().")