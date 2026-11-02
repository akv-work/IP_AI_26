

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET_URL = "https://universe.roboflow.com/mohamed-traore-2ekkp/cats-n9b87/dataset/3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Открытые изображения PyTorch Hub и COCO. Не включаются в тестовую метрику.
EXAMPLES = [
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def dependencies():
    try:
        import torch
        import ultralytics
        import yaml
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Установите зависимости: python -m pip install ultralytics==8.4.14 roboflow") from exc
    return torch, ultralytics, yaml, Image


def extract_archive(archive, destination):
    """Распаковка ZIP без возможности записи за пределы целевой папки."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Недопустимый путь в ZIP: {info.filename}")
        z.extractall(destination)
    configs = sorted(destination.rglob("data.yaml"))
    if len(configs) != 1:
        raise ValueError("В архиве должен быть ровно один data.yaml.")
    return configs[0]


def get_data(args):
    if args.data:
        return Path(args.data).expanduser().resolve()
    if args.archive:
        return extract_archive(args.archive, BASE / "data" / "cats-v3")
    candidates = [BASE / "data/cats-v3/data.yaml", BASE / "Cats-3/data.yaml"]
    for path in candidates:
        if path.is_file():
            return path
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Нужен Cats v3: укажите --data data.yaml, --archive ZIP "
                           "или переменную ROBOFLOW_API_KEY. Страница: " + DATASET_URL)
    try:
        from roboflow import Roboflow
        dataset = (Roboflow(api_key=key).workspace("mohamed-traore-2ekkp")
                   .project("cats-n9b87").version(3)
                   .download("yolov8", location=str(BASE / "data/cats-v3")))
    except Exception:
        # Не выводим исключения SDK: в них может находиться URL с API-ключом.
        raise RuntimeError("Не удалось загрузить Cats v3 из Roboflow. Проверьте доступ "
                           "или скачайте ZIP вручную и задайте --archive.") from None
    return Path(dataset.location) / "data.yaml"


def prepare_data(source, out):
    """Проверить разметку, сохранить официальные split и исправить пути экспорта."""
    _, _, yaml, Image = dependencies()
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not isinstance(names, list) or [str(n).lower() for n in names] != ["cat"]:
        raise ValueError(f"Ожидался один класс cat, получено: {names!r}")
    normalized = {"names": ["cat"], "nc": 1}
    audit = {"source": DATASET_URL, "splits": {}, "duplicates_within_split": {}}
    hashes, stems = {}, {}
    for split, alias in [("train", "train"), ("val", "valid"), ("test", "test")]:
        raw = config.get(split)
        if not isinstance(raw, str):
            raise ValueError(f"В data.yaml отсутствует каталог {split}.")
        root = Path(config.get("path", source.parent))
        if not root.is_absolute():
            root = source.parent / root
        options = [root / raw, source.parent / raw,
                   source.parent / alias / "images", source.parent / split / "images"]
        directory = next((p.resolve() for p in options if p.is_dir()), None)
        if directory is None:
            raise FileNotFoundError(f"Не найден каталог изображений {split}: {raw}")
        images = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Пустая выборка: {split}")
        label_dir = directory.parent / "labels"
        boxes = backgrounds = duplicate_count = 0
        for image in images:
            label = label_dir / image.relative_to(directory).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Нет разметки: {label}; для фона нужен пустой TXT.")
            with Image.open(image) as im:
                im.verify()
            lines = label.read_text(encoding="utf-8").splitlines()
            objects = 0
            for line in lines:
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Ожидалось class xc yc w h: {label}")
                cls, x, y, w, h = map(float, fields)
                if not all(math.isfinite(v) for v in [cls, x, y, w, h]):
                    raise ValueError(f"NaN/Inf в {label}")
                if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Некорректные координаты или класс: {label}")
                if min(x-w/2, y-h/2) < -0.002 or max(x+w/2, y+h/2) > 1.002:
                    raise ValueError(f"Рамка за пределами изображения: {label}")
                objects += 1
            boxes += objects
            backgrounds += objects == 0
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            # Roboflow добавляет .rf.<hash>; исходное имя выявляет варианты одного фото.
            stem = image.stem.split(".rf.")[0]
            for registry, key in [(hashes, digest), (stems, stem)]:
                if key in registry and registry[key] != split:
                    raise ValueError(f"Пересечение {registry[key]} и {split}: {image.name}")
            duplicate_count += digest in hashes
            hashes[digest], stems[stem] = split, split
        normalized[split] = directory.as_posix()
        audit["splits"][split] = {"images": len(images), "boxes": boxes, "backgrounds": backgrounds}
        audit["duplicates_within_split"][split] = duplicate_count
    save_json(out / "dataset_audit.json", audit)
    prepared = out / "data.yaml"
    prepared.write_text(yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps(audit["splits"], ensure_ascii=False, indent=2))
    return prepared


def evaluate(weights, data, out, args):
    _, ultralytics, _, _ = dependencies()
    model = ultralytics.YOLO(str(weights))
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=0,
                        conf=0.001, iou=0.7, plots=True, save_json=True,
                        project=str(out), name="test", exist_ok=False)
    result = {"weights": str(Path(weights).resolve()), "split": "test",
              "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "evaluation_conf": 0.001, "iou_parameter": 0.7,
              "note": "Precision и Recall рассчитаны библиотекой в рабочей точке PR-кривой; "
                      "порог вывода фотографий не применяется при вычислении mAP."}
    save_json(out / "test_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def report_figures(weights, data, out, args):
    """Графики по реальному журналу и первые шесть test без отбора по качеству."""
    _, ultralytics, yaml, Image = dependencies()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    history = out / "train" / "results.csv"
    if history.is_file():
        with history.open(encoding="utf-8") as file:
            rows = [{k.strip(): float(v) for k, v in row.items()}
                    for row in csv.DictReader(file)]
        epochs = [r["epoch"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for key, label in [("train/box_loss", "Train box"), ("val/box_loss", "Val box")]:
            axes[0].plot(epochs, [r[key] for r in rows], label=label)
        for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
            axes[1].plot(epochs, [r[key] for r in rows], label=label)
        for ax in axes:
            ax.set_xlabel("Эпоха"); ax.grid(alpha=.25); ax.legend()
        axes[0].set_ylabel("Потери локализации")
        axes[1].set_ylabel("mAP на validation"); axes[1].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / "training.png", dpi=180); plt.close(fig)
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
            if not line.strip(): continue
            _, x, y, bw, bh = map(float, line.split())
            draw.rectangle(((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h), outline="lime", width=3)
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf,
                               device=args.device, verbose=False)[0]
        for box in result.boxes:
            xy = box.xyxy[0].tolist()
            draw.rectangle(xy, outline="red", width=2)
            draw.text((xy[0]+3, xy[1]+3), f"cat {box.conf.item():.2f}", fill="red", stroke_width=1, stroke_fill="white")
        canvas.paste(im.resize((416, 416)), ((i%3)*416, (i//3)*446))
        ImageDraw.Draw(canvas).text(((i%3)*416+5, (i//3)*446+420), path.name[:48], fill="black")
    canvas.save(out / "test_examples.jpg")


def predict(weights, sources, out, args):
    _, ultralytics, _, Image = dependencies()
    model = ultralytics.YOLO(str(weights))
    rows = []
    for index, source in enumerate(sources, 1):
        results = model.predict(source=str(source), imgsz=args.imgsz, device=args.device,
                                conf=args.conf, iou=0.7, save=False, verbose=False)
        for j, result in enumerate(results, 1):
            output = out / f"prediction_{index:02d}_{j:02d}.jpg"
            result.save(filename=str(output))
            boxes = [{"class": int(b.cls.item()), "name": result.names[int(b.cls.item())],
                      "confidence": float(b.conf.item()),
                      "xyxy": [float(v) for v in b.xyxy[0].tolist()]} for b in result.boxes]
            rows.append({"source": str(source), "output": str(output), "boxes": boxes})
            print(f"{source}: обнаружено {len(boxes)}; {output}")
            if args.show:
                with Image.open(output) as im:
                    im.show()
    save_json(out / "predictions.json", rows)
    return rows


def internet_examples(out):
    """Скачать демонстрационные фото; сохранить источник каждого файла."""
    paths, records = [], []
    folder = out / "internet"
    folder.mkdir(exist_ok=True)
    for i, url in enumerate(EXAMPLES, 1):
        path = folder / f"example_{i}.jpg"
        try:
            if not path.is_file():
                request = urllib.request.Request(url, headers={"User-Agent": "Lab3 educational example"})
                with urllib.request.urlopen(request, timeout=15) as response:
                    path.write_bytes(response.read())
            from PIL import Image
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            print(f"Не удалось скачать пример {url}: {type(exc).__name__}", file=sys.stderr)
            continue
        paths.append(path)
        records.append({"url": url, "file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    save_json(out / "internet_sources.json", records)
    return paths


def train(data, out, args):
    torch, ultralytics, _, _ = dependencies()
    model = ultralytics.YOLO(str(BASE / "yolo26s.pt"))
    started = time.perf_counter()
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
                seed=16, deterministic=True, patience=args.epochs,
                mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                amp=torch.cuda.is_available() and args.device != "cpu",
                plots=True, project=str(out), name="train", exist_ok=False)
    weights = Path(model.trainer.best)
    if not weights.is_file():
        raise RuntimeError("После обучения не найден best.pt")
    save_json(out / "run_info.json", {
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "seed": 16,
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds_train_and_val": time.perf_counter() - started,
        "epochs_requested": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "weights": str(weights), "dataset": DATASET_URL,
        "selection": "best.pt выбирается Ultralytics по validation fitness, test не участвует"})
    return weights


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=["all", "prepare", "train", "evaluate", "predict"], default="all")
    p.add_argument("--data", help="Путь к data.yaml экспорта Cats v3")
    p.add_argument("--archive", help="ZIP-архив экспорта Cats v3 в формате YOLO")
    p.add_argument("--weights", type=Path, help="best.pt для evaluate/predict")
    p.add_argument("--source", nargs="+", help="Изображения или URL для предсказания")
    p.add_argument("--out", type=Path, default=BASE / "results_lab3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="0 для первой GPU, cpu для процессора")
    p.add_argument("--conf", type=float, default=0.25, help="Порог только для визуализации")
    p.add_argument("--show", action="store_true", help="Открыть результаты распознавания")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32 or not 0 < args.conf < 1:
        p.error("epochs/batch должны быть > 0, imgsz — кратным 32, 0 < conf < 1")
    torch, _, _, _ = dependencies()
    args.device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    out = args.out.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == "predict":
        if not args.weights or not args.weights.is_file():
            p.error("Для predict требуется --weights с существующим best.pt")
        sources = args.source
        if not sources:
            from tkinter import Tk, filedialog
            root = Tk(); root.withdraw()
            sources = list(filedialog.askopenfilenames(title="Выберите фотографии котов"))
            root.destroy()
        if sources:
            predict(args.weights, sources, out, args)
        return
    data = prepare_data(get_data(args), out)
    if args.mode == "prepare":
        return
    if args.mode == "evaluate":
        if not args.weights or not args.weights.is_file():
            p.error("Для evaluate требуется --weights с существующим best.pt")
        evaluate(args.weights, data, out, args)
        return
    weights = train(data, out, args)
    if args.mode == "all":
        evaluate(weights, data, out, args)
        report_figures(weights, data, out, args)
        sources = args.source or internet_examples(out)
        if not sources:
            raise RuntimeError("Обучение и тест готовы, но интернет-фото не загрузились. "
                               "Выполните predict --weights ... --source ...")
        predict(weights, sources, out, args)
    print(f"Готово. Результаты: {out}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
