

import os
import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO


DATASET_DIR = "License-Plate-Recognition-11"
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
WEIGHTS = None  # None = найти автоматически
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "test", "images")
OUTPUT_DIR = "evaluation"
IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = 0



def find_best_weights():
    if WEIGHTS and os.path.exists(WEIGHTS):
        return WEIGHTS
    candidates = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
    if not candidates:
        raise FileNotFoundError("Не найдены веса best.pt. Обучите модель.")
    return max(candidates, key=os.path.getmtime)


def plot_confidence_histogram(model, images, out_dir):

    confs = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        r = model(img, conf=0.01, iou=IOU, device=DEVICE, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            confs.extend(r.boxes.conf.cpu().numpy().tolist())

    if not confs:
        print("[WARN] Нет детекций для гистограммы уверенности")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confs, bins=40, color="#2E86AB", edgecolor="black", alpha=0.85)
    ax.axvline(CONF, color="red", linestyle="--", linewidth=2,
               label=f"Порог = {CONF}")
    ax.set_xlabel("Уверенность (confidence)", fontsize=12)
    ax.set_ylabel("Количество детекций", fontsize=12)
    ax.set_title("Распределение уверенности детекций", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_histogram.png"), dpi=150)
    plt.close()
    print(f"  ✓ confidence_histogram.png ({len(confs)} детекций)")


def plot_detections_per_image(model, images, out_dir):

    counts = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        r = model(img, conf=CONF, iou=IOU, device=DEVICE, verbose=False)[0]
        counts.append(len(r.boxes) if r.boxes is not None else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    counter = Counter(counts)
    ks = sorted(counter.keys())
    vs = [counter[k] for k in ks]
    bars = ax.bar(ks, vs, color="#A23B72", edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                str(v), ha="center", fontsize=10)
    ax.set_xlabel("Количество найденных номеров на изображении", fontsize=12)
    ax.set_ylabel("Количество изображений", fontsize=12)
    ax.set_title("Распределение числа детекций по изображениям",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detections_per_image.png"), dpi=150)
    plt.close()
    print(f"  ✓ detections_per_image.png (обработано {len(counts)} фото)")


def plot_box_sizes(model, images, out_dir):

    widths, heights = [], []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model(img, conf=CONF, iou=IOU, device=DEVICE, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xyxy:
                widths.append((x2 - x1) / w * 100)   # % от ширины
                heights.append((y2 - y1) / h * 100)  # % от высоты

    if not widths:
        print("[WARN] Нет боксов для анализа размеров")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(widths, heights, alpha=0.5, s=25, c="#F18F01",
                    edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Ширина бокса, % от ширины изображения", fontsize=12)
    ax.set_ylabel("Высота бокса, % от высоты изображения", fontsize=12)
    ax.set_title("Распределение размеров обнаруженных номеров",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_sizes.png"), dpi=150)
    plt.close()
    print(f"  ✓ box_sizes.png ({len(widths)} боксов)")


def plot_iou_distribution(model, images, out_dir, max_imgs=200):

    from ultralytics.utils.metrics import box_iou
    import torch

    ious = []
    labels_dir = os.path.join(DATASET_DIR, "test", "labels")

    for img_path in images[:max_imgs]:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(labels_dir,
                                  os.path.basename(img_path).rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        # GT
        gts = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gts.append([x1, y1, x2, y2])

        if not gts:
            continue

        r = model(img, conf=CONF, iou=IOU, device=DEVICE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        preds = r.boxes.xyxy.cpu().numpy()

        # Матрица IoU: preds × gts → берём максимум для каждого GT
        iou_matrix = box_iou(torch.tensor(preds), torch.tensor(gts)).numpy()
        for j in range(len(gts)):
            ious.append(iou_matrix[:, j].max())

    if not ious:
        print("[WARN] Не удалось посчитать IoU")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ious, bins=30, color="#3BB273", edgecolor="black", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2,
               label="Порог IoU = 0.5")
    mean_iou = np.mean(ious)
    ax.axvline(mean_iou, color="blue", linestyle="--", linewidth=2,
               label=f"Средний IoU = {mean_iou:.3f}")
    ax.set_xlabel("IoU с эталонной рамкой", fontsize=12)
    ax.set_ylabel("Количество объектов", fontsize=12)
    ax.set_title("Распределение IoU предсказаний и эталонов",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_distribution.png"), dpi=150)
    plt.close()
    print(f"  ✓ iou_distribution.png (средний IoU = {mean_iou:.3f})")


def plot_pr_f1_curves(model, out_dir):

    metrics = model.val(
        data=DATA_YAML, split="test", imgsz=IMGSZ, device=DEVICE,
        plots=False, verbose=False,
    )

    # PR-кривая
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- PR ---
    try:
        px = metrics.curves_results[0][0]   # x: recall
        py = metrics.curves_results[0][1]   # y: precision
        axes[0].plot(px, py[0] if py.ndim > 1 else py,
                     color="#2E86AB", linewidth=2)
        axes[0].fill_between(px, 0, py[0] if py.ndim > 1 else py,
                             alpha=0.2, color="#2E86AB")
        axes[0].set_xlabel("Recall", fontsize=12)
        axes[0].set_ylabel("Precision", fontsize=12)
        axes[0].set_title(f"PR-кривая (mAP@0.5 = {metrics.box.map50:.4f})",
                          fontsize=13, fontweight="bold")
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.05)
    except Exception as e:
        print(f"[WARN] PR-кривая: {e}")

    # --- F1 ---
    try:
        px = metrics.curves_results[2][0]   # x: confidence
        py = metrics.curves_results[2][1]   # y: F1
        f1 = py[0] if py.ndim > 1 else py
        best_idx = np.argmax(f1)
        axes[1].plot(px, f1, color="#C73E1D", linewidth=2)
        axes[1].fill_between(px, 0, f1, alpha=0.2, color="#C73E1D")
        axes[1].axvline(px[best_idx], color="green", linestyle="--",
                        label=f"Оптимум: conf={px[best_idx]:.2f}, F1={f1[best_idx]:.3f}")
        axes[1].set_xlabel("Confidence", fontsize=12)
        axes[1].set_ylabel("F1-score", fontsize=12)
        axes[1].set_title("F1-кривая", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1.05)
    except Exception as e:
        print(f"[WARN] F1-кривая: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_f1_curves.png"), dpi=150)
    plt.close()
    print("  ✓ pr_f1_curves.png")


def plot_inference_time(model, images, out_dir, n=100):

    times = []
    for img_path in images[:n]:
        img = cv2.imread(img_path)
        if img is None:
            continue
        t0 = time.perf_counter()
        model(img, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)  # мс

    if not times:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(times, bins=30, color="#8E5572", edgecolor="black", alpha=0.85)
    mean_t = np.mean(times)
    ax.axvline(mean_t, color="red", linestyle="--", linewidth=2,
               label=f"Среднее = {mean_t:.1f} мс")
    ax.set_xlabel("Время инференса, мс", fontsize=12)
    ax.set_ylabel("Количество изображений", fontsize=12)
    ax.set_title(f"Скорость инференса (~{1000/mean_t:.0f} FPS)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_time.png"), dpi=150)
    plt.close()
    print(f"  ✓ inference_time.png (среднее {mean_t:.1f} мс, ~{1000/mean_t:.0f} FPS)")


def plot_loss_curves(out_dir):

    csv_files = glob.glob("runs/detect/**/results.csv", recursive=True)
    if not csv_files:
        print("[WARN] results.csv не найден — пропуск loss-кривых")
        return

    csv_path = max(csv_files, key=os.path.getmtime)
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    # Извлекаем колонки box_loss, cls_loss, dfl_loss + их val-версии
    headers = open(csv_path).readline().strip().split(",")
    epochs = data[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss", "#2E86AB"),
        ("train/cls_loss", "val/cls_loss", "Class Loss", "#C73E1D"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss", "#3BB273"),
    ]

    for ax, (tr, vl, title, color) in zip(axes, loss_pairs):
        if tr in headers and vl in headers:
            i_tr = headers.index(tr)
            i_vl = headers.index(vl)
            ax.plot(epochs, data[:, i_tr], color=color, linewidth=2, label="Train")
            ax.plot(epochs, data[:, i_vl], color=color, linewidth=2,
                    linestyle="--", alpha=0.7, label="Val")
            ax.set_xlabel("Эпоха", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)

    plt.suptitle("Кривые функции потерь", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print("  ✓ loss_curves.png")



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weights = find_best_weights()
    print(f"[INFO] Модель: {weights}\n")

    model = YOLO(weights)

    images = sorted(
        glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
        + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
        + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpeg"))
    )
    print(f"[INFO] Тестовых изображений: {len(images)}\n")
    print("[INFO] Генерация графиков...")

    # 1. Loss-кривые из истории обучения
    plot_loss_curves(OUTPUT_DIR)

    # 2. PR + F1 кривые
    plot_pr_f1_curves(model, OUTPUT_DIR)

    # 3. Гистограмма уверенности
    plot_confidence_histogram(model, images, OUTPUT_DIR)

    # 4. Детекций на изображение
    plot_detections_per_image(model, images, OUTPUT_DIR)

    # 5. Размеры боксов
    plot_box_sizes(model, images, OUTPUT_DIR)

    # 6. IoU с эталонами
    plot_iou_distribution(model, images, OUTPUT_DIR)

    # 7. Время инференса
    plot_inference_time(model, images, OUTPUT_DIR)

    print(f"\n[OK] Все графики сохранены в: {OUTPUT_DIR}/")
    print("\nЧто генерируется:")
    print("  • loss_curves.png            — кривые box/cls/dfl loss")
    print("  • pr_f1_curves.png           — PR-кривая и F1-кривая")
    print("  • confidence_histogram.png   — распределение уверенности")
    print("  • detections_per_image.png   — сколько номеров на фото")
    print("  • box_sizes.png              — размеры боксов")
    print("  • iou_distribution.png       — IoU с эталоном")
    print("  • inference_time.png         — скорость (FPS)")


if __name__ == "__main__":
    main()