import json
import zipfile
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def extract_archive(archive, destination):
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


def dependencies():
    try:
        import torch
        import ultralytics
        import yaml
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Установите зависимости: python -m pip install ultralytics==8.4.14 roboflow"
        ) from exc
    return torch, ultralytics, yaml, Image
