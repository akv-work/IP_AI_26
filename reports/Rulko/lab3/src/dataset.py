from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from config import BASE, DATASET_URL, IMAGE_EXTENSIONS
from utils.io import dependencies, extract_archive, save_json


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
        raise RuntimeError(
            "Нужен Cats v3: укажите --data data.yaml, --archive ZIP "
            "или переменную ROBOFLOW_API_KEY. Страница: " + DATASET_URL
        )
    try:
        from roboflow import Roboflow
        dataset = (
            Roboflow(api_key=key)
            .workspace("mohamed-traore-2ekkp")
            .project("cats-n9b87")
            .version(3)
            .download("yolov8", location=str(BASE / "data/cats-v3"))
        )
    except Exception:
        raise RuntimeError(
            "Не удалось загрузить Cats v3 из Roboflow. Проверьте доступ "
            "или скачайте ZIP вручную и задайте --archive."
        ) from None
    return Path(dataset.location) / "data.yaml"


def prepare_data(source, out):
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
        options = [
            root / raw,
            source.parent / raw,
            source.parent / alias / "images",
            source.parent / split / "images",
        ]
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
                if min(x - w / 2, y - h / 2) < -0.002 or max(x + w / 2, y + h / 2) > 1.002:
                    raise ValueError(f"Рамка за пределами изображения: {label}")
                objects += 1
            boxes += objects
            backgrounds += objects == 0
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
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
