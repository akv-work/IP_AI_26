from __future__ import annotations

import platform
import time
from pathlib import Path

from config import BASE, DATASET_URL
from utils.io import dependencies, save_json


def train(data, out, args):
    torch, ultralytics, _, _ = dependencies()
    model = ultralytics.YOLO(str(BASE / "yolo26s.pt"))
    started = time.perf_counter()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        seed=16,
        deterministic=True,
        patience=args.epochs,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        amp=torch.cuda.is_available() and args.device != "cpu",
        plots=True,
        project=str(out),
        name="train",
        exist_ok=False,
    )
    weights = Path(model.trainer.best)
    if not weights.is_file():
        raise RuntimeError("После обучения не найден best.pt")
    save_json(out / "run_info.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "seed": 16,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seconds_train_and_val": time.perf_counter() - started,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "weights": str(weights),
        "dataset": DATASET_URL,
        "selection": "best.pt выбирается Ultralytics по validation fitness, test не участвует",
    })
    return weights
