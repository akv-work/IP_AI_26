from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from config import BASE
from data.dataset import get_data, prepare_data
from evaluate import evaluate, report_figures
from predict import internet_examples, predict
from train import train
from utils.io import dependencies


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p = build_parser()
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
            root = Tk()
            root.withdraw()
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
            raise RuntimeError(
                "Обучение и тест готовы, но интернет-фото не загрузились. "
                "Выполните predict --weights ... --source ..."
            )
        predict(weights, sources, out, args)

    print(f"Готово. Результаты: {out}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
