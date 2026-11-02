import io
import random
from pathlib import Path
from roboflow import Roboflow
import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from ultralytics import YOLO

random.seed(42)
print("GPU доступен:", torch.cuda.is_available())


ROBOFLOW_API_KEY = "QUYMcDyS147x6eWX6IUw"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(11).download("yolov8")

DATASET_DIR = Path(dataset.location)
print("Датасет скачан в:", DATASET_DIR)


yaml_path = DATASET_DIR / "data.yaml"
with open(yaml_path) as f:
    data_cfg = yaml.safe_load(f)

print("Классы:", data_cfg["names"], "| число классов:", len(data_cfg["names"]))

data_cfg["path"] = str(DATASET_DIR)
data_cfg["train"] = "train/images"
data_cfg["val"] = "valid/images"
data_cfg["test"] = "test/images"
with open(yaml_path, "w") as f:
    yaml.safe_dump(data_cfg, f, allow_unicode=True)

for split in ["train", "valid", "test"]:
    folder = DATASET_DIR / split / "images"
    n = len(list(folder.glob("*"))) if folder.exists() else 0
    print(f"{split:>6}: {n} изображений")

def show_samples(split="train", n=6):
    img_paths = random.sample(list((DATASET_DIR / split / "images").glob("*")), n)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, p in zip(axes.flatten(), img_paths):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        ax.imshow(img)
        ax.axis("off")
        lbl = DATASET_DIR / split / "labels" / (p.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                ax.add_patch(patches.Rectangle((x0, y0), w * W, h * H,
                                               fill=False, edgecolor="lime", linewidth=2))
    plt.suptitle("Примеры из обучающей выборки (зелёным показана разметка)")
    plt.tight_layout()
    plt.show()

show_samples("train")


model = YOLO("yolo12n.pt")

EPOCHS = 20   

results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=640,
    batch=16,
    project="runs_plates",
    name="yolo12n",
    exist_ok=True,
    seed=42,
)

save_dir = Path(model.trainer.save_dir)
print("Результаты обучения в:", save_dir)


df = pd.read_csv(save_dir / "results.csv")
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(df["epoch"], df["train/box_loss"], label="Train box loss", marker="o", markersize=3)
axes[0].plot(df["epoch"], df["val/box_loss"], label="Val box loss", marker="o", markersize=3)
axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
axes[0].set_title("YOLO12n: изменение ошибки"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", marker="o", markersize=3)
axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", marker="o", markersize=3)
axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("mAP")
axes[1].set_title("YOLO12n: изменение mAP на валидации"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


best_model = YOLO(str(save_dir / "weights" / "best.pt"))
metrics = best_model.val(data=str(yaml_path), split="test", imgsz=640)

print("Результаты на тестовой выборке:")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


def load_image(src):
    return Image.open(src).convert("RGB")

def detect_and_show(sources, conf=0.25):
    for src in sources:
        img = load_image(src)
        res = best_model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        plotted = res.plot()[..., ::-1]   
        n = len(res.boxes)
        confs = ", ".join(f"{c:.2f}" for c in res.boxes.conf.tolist())

        plt.figure(figsize=(9, 6))
        plt.imshow(plotted)
        plt.axis("off")
        plt.title(f"Найдено номеров: {n}" + (f" (уверенность: {confs})" if n else ""))
        plt.show()

sources = ["car.jpg"]
detect_and_show(sources)