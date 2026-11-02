from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from config import EXAMPLES
from utils.io import dependencies, save_json


def predict(weights, sources, out, args):
    _, ultralytics, _, Image = dependencies()
    model = ultralytics.YOLO(str(weights))
    rows = []
    for index, source in enumerate(sources, 1):
        results = model.predict(
            source=str(source),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=0.7,
            save=False,
            verbose=False,
        )
        for j, result in enumerate(results, 1):
            output = out / f"prediction_{index:02d}_{j:02d}.jpg"
            result.save(filename=str(output))
            boxes = [
                {
                    "class": int(b.cls.item()),
                    "name": result.names[int(b.cls.item())],
                    "confidence": float(b.conf.item()),
                    "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                }
                for b in result.boxes
            ]
            rows.append({"source": str(source), "output": str(output), "boxes": boxes})
            print(f"{source}: обнаружено {len(boxes)}; {output}")
            if args.show:
                with Image.open(output) as im:
                    im.show()
    save_json(out / "predictions.json", rows)
    return rows


def internet_examples(out):
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
        records.append({
            "url": url,
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    save_json(out / "internet_sources.json", records)
    return paths
