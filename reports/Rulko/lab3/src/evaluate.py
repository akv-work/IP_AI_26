from __future__ import annotations

import csv
import json
from pathlib import Path

from config import IMAGE_EXTENSIONS
from utils.io import dependencies, save_json


def evaluate(weights, data, out, args):
    _, ultralytics, _, _ = dependencies()
    model = ultralytics.YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        save_json=True,
        project=str(out),
        name="test",
        exist_ok=False,
    )
    result = {
        "weights": str(Path(weights).resolve()),
        "split": "test",
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "evaluation_conf": 0.001,
        "iou_parameter": 0.7,
        "note": (
            "Precision и Recall рассчитаны библиотекой в рабочей точке PR-кривой; "
            "порог вывода фотографий не применяется при вычислении mAP."
        ),
    }
    save_json(out / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def report_figures(weights, data, out, args):
    _, ultralytics, yaml, Image = dependencies()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    history = out / "train" / "results.csv"
    if history.is_file():
        with history.open(encoding="utf-8") as file:
            rows = [
                {k.strip(): float(v) for k, v in row.items()}
                for row in csv.DictReader(file)
            ]
        epochs = [r["epoch"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for key, label in [("train/box_loss", "Train box"), ("val/box_loss", "Val box")]:
            axes[0].plot(epochs, [r[key] for r in rows], label=label)
        for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
            axes[1].plot(epochs, [r[key] for r in rows], label=label)
        for ax in axes:
            ax.set_xlabel("Эпоха")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("Потери локализации")
        axes[1].set_ylabel("mAP на validation")
        axes[1].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "training.png", dpi=180)
        plt.close(fig)

    cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    directory = Path(cfg["test"])
    files = sorted(p for p in directory.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:6]
    model = ultralytics.YOLO(str(weights))
    canvas = Image.new("RGB", (3 * 416, 2 * 446), "white")

    for i, path in enumerate(files):
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        w, h = im.size
        label = directory.parent / "labels" / path.with_suffix(".txt").name
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            _, x, y, bw, bh = map(float, line.split())
            draw.rectangle(
                ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h),
                outline="lime",
                width=3,
            )
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        for box in result.boxes:
            xy = box.xyxy[0].tolist()
            draw.rectangle(xy, outline="red", width=2)
            draw.text(
                (xy[0] + 3, xy[1] + 3),
                f"cat {box.conf.item():.2f}",
                fill="red",
                stroke_width=1,
                stroke_fill="white",
            )
        canvas.paste(im.resize((416, 416)), ((i % 3) * 416, (i // 3) * 446))
        ImageDraw.Draw(canvas).text(
            ((i % 3) * 416 + 5, (i // 3) * 446 + 420),
            path.name[:48],
            fill="black",
        )
    canvas.save(out / "test_examples.jpg")
