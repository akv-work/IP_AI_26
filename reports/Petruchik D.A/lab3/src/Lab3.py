import sys

for name in ("prefix", "exec_prefix", "base_prefix", "base_exec_prefix"):
    value = getattr(sys, name)
    if len(value) == 2 and value[1] == ":":
        setattr(sys, name, value + "\\")

import io
import os
import random
from getpass import getpass
from pathlib import Path

import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps

import torch
from roboflow import Roboflow
from ultralytics import YOLO

SEED = 42
EPOCHS = 20
IMGSZ = 640
BATCH = 8  
DEVICE = 0 if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

print("GPU доступен:", torch.cuda.is_available())
print("Устройство:", DEVICE)

api_key = os.getenv("ROBOFLOW_API_KEY") or getpass(
    "Введите API-ключ Roboflow: "
)

rf = Roboflow(api_key=api_key)
project = rf.workspace("mohamed-traore-2ekkp").project("cats-n9b87")
dataset = project.version(3).download("yolov8")

DATASET_DIR = Path(dataset.location).resolve()
print("Датасет скачан в:", DATASET_DIR)

yaml_path = DATASET_DIR / "data.yaml"

with yaml_path.open(encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

print("Классы:", data_cfg["names"])
print("Число классов:", len(data_cfg["names"]))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(split):
    folder = DATASET_DIR / split / "images"
    return sorted(
        p for p in folder.glob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


for split in ["train", "valid", "test"]:
    paths = image_paths(split)
    print(f"{split:>6}: {len(paths)} изображений")

    if not paths:
        raise RuntimeError(
            f"Выборка {split} отсутствует или пуста: "
            f"{DATASET_DIR / split / 'images'}"
        )

data_cfg["path"] = str(DATASET_DIR)
data_cfg["train"] = "train/images"
data_cfg["val"] = "valid/images"
data_cfg["test"] = "test/images"

with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data_cfg, f, allow_unicode=True)

def class_name(class_id):
    names = data_cfg["names"]
    if isinstance(names, dict):
        return names.get(class_id, names.get(str(class_id), str(class_id)))
    return names[class_id]


def show_samples(split="train", n=6):
    paths = image_paths(split)
    if not paths or n < 1:
        return

    selected = random.sample(paths, min(n, len(paths)))
    rows = (len(selected) + 2) // 3

    fig, axes = plt.subplots(
        rows, 3, figsize=(14, 4 * rows), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, path in zip(axes.flat, selected):
        with Image.open(path) as source:
            img = source.convert("RGB")

        width, height = img.size
        ax.imshow(img)

        label_path = DATASET_DIR / split / "labels" / f"{path.stem}.txt"

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue

                cls, cx, cy, w, h = map(float, parts)
                x0 = (cx - w / 2) * width
                y0 = (cy - h / 2) * height

                ax.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        w * width,
                        h * height,
                        fill=False,
                        edgecolor="lime",
                        linewidth=2,
                    )
                )

                ax.text(
                    x0, max(0, y0 - 3),
                    class_name(int(cls)),
                    color="black",
                    fontsize=9,
                    bbox=dict(facecolor="lime", alpha=0.7, pad=1),
                )

    fig.suptitle(f"Коты: примеры выборки {split} с разметкой")
    plt.tight_layout()
    plt.show()


show_samples("train")

model = YOLO("yolo12s.pt")

results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=0,
    project="runs_cats",
    name="yolo12s_variant10",
    exist_ok=False,
    seed=SEED,
    plots=True,
)

save_dir = Path(model.trainer.save_dir)
print("Результаты обучения в:", save_dir)

df = pd.read_csv(save_dir / "results.csv")
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    df["epoch"], df["train/box_loss"],
    label="Train box loss", marker="o", markersize=3,
)
axes[0].plot(
    df["epoch"], df["val/box_loss"],
    label="Val box loss", marker="o", markersize=3,
)
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Loss")
axes[0].set_title("YOLO12s: ошибка локализации")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    df["epoch"], df["metrics/mAP50(B)"],
    label="mAP@0.5", marker="o", markersize=3,
)
axes[1].plot(
    df["epoch"], df["metrics/mAP50-95(B)"],
    label="mAP@0.5:0.95", marker="o", markersize=3,
)
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("mAP")
axes[1].set_title("YOLO12s: mAP на валидации")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / "training_curves.png", dpi=200)
plt.show()

best_model = YOLO(str(save_dir / "weights" / "best.pt"))

metrics = best_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=0,
    project=str(save_dir),
    name="test_evaluation",
    plots=True,
)

test_results = {
    "Precision": float(metrics.box.mp),
    "Recall": float(metrics.box.mr),
    "mAP@0.5": float(metrics.box.map50),
    "mAP@0.5:0.95": float(metrics.box.map),
}

print("\nРезультаты на тестовой выборке:")
for name, value in test_results.items():
    print(f"{name}: {value:.4f}")

pd.DataFrame([test_results]).to_csv(
    save_dir / "test_metrics.csv", index=False
)

def load_image(src):
    src = str(src)

    if src.startswith(("http://", "https://")):
        response = requests.get(src, timeout=30)
        response.raise_for_status()
        with Image.open(io.BytesIO(response.content)) as img:
            return ImageOps.exif_transpose(img).convert("RGB")

    with Image.open(src) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def detect_and_show(sources, conf=0.25, prefix="external"):
    output_dir = save_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sources, start=1):
        img = load_image(src)

        result = best_model.predict(
            source=img,
            conf=conf,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )[0]

        plotted = result.plot()[..., ::-1]
        count = len(result.boxes)
        confidences = ", ".join(
            f"{c:.2f}" for c in result.boxes.conf.tolist()
        )

        Image.fromarray(plotted).save(
            output_dir / f"{prefix}_{i:02d}.jpg"
        )

        plt.figure(figsize=(9, 6))
        plt.imshow(plotted)
        plt.axis("off")
        plt.title(
            f"Найдено котов: {count}"
            + (f" (уверенность: {confidences})" if count else "")
        )
        plt.tight_layout()
        plt.show()



detect_and_show(image_paths("test")[:3], prefix="test")


sources = [
    r"D:\cats\cat1.jpg",
    r"D:\cats\cat2.jpg",
    r"D:\cats\cat3.jpg",
]


if sources:
    detect_and_show(sources, prefix="internet")
else:
    print(
        "Фото из интернета не загружены, загрузите фото"
    
    )