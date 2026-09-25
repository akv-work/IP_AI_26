
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

BASE = Path(__file__).resolve().parent


def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # CrossEntropyLoss получает логиты, поэтому Softmax здесь не нужен.
        return self.classifier(self.features(x))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    labels, predictions, probabilities = [], [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        predicted = logits.argmax(1)
        loss_sum += criterion(logits, targets).item()
        correct += (predicted == targets).sum().item()
        count += targets.size(0)
        labels.extend(targets.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())
        probabilities.extend(logits.softmax(1).max(1).values.cpu().tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "correct": correct, "total": count, "errors": count - correct,
            "labels": labels, "predictions": predictions,
            "confidence": probabilities}


def save_plots(history, result, test, out):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for prefix, label in [("train", "Обучение"), ("val", "Валидация")]:
        axes[0].plot(epochs, [r[prefix + "_loss"] for r in history], "o-", label=label)
        axes[1].plot(epochs, [100 * (1 - r[prefix + "_accuracy"]) for r in history],
                     "o-", label=label)
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Средняя CrossEntropyLoss")
    axes[1].set_ylabel("Ошибка классификации, %")
    fig.savefig(out / "learning_curves.png", dpi=200)
    plt.close(fig)

    matrix = np.zeros((10, 10), dtype=int)
    for actual, predicted in zip(result["labels"], result["predictions"]):
        matrix[actual, predicted] += 1
    np.savetxt(out / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(6.3, 5.3), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Предсказанная цифра",
           ylabel="Истинная цифра")
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    for name, indices, title in [
        ("test_examples", list(range(12)), "Первые 12 изображений тестовой выборки"),
        ("test_errors", [i for i, (a, p) in enumerate(zip(result["labels"], result["predictions"]))
                         if a != p][:12], "Первые ошибки в порядке тестовой выборки"),
    ]:
        fig, axes = plt.subplots(2, 6, figsize=(10, 3.8), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, indices):
            ax.imshow(test.data[index], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f'№{index}: {result["labels"][index]} → {result["predictions"][index]}\n'
                         f'p = {100 * result["confidence"][index]:.1f}%', fontsize=10)
        fig.suptitle(title, fontsize=12)
        fig.savefig(out / (name + ".png"), dpi=200)
        plt.close(fig)


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Обучение CNN на MNIST с RMSprop")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size и lr должны быть положительными")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    full_train = MNIST(args.data_dir, train=True, download=True, transform=image_transform())
    train, val = random_split(full_train, [55000, 5000],
                             generator=torch.Generator().manual_seed(args.seed))
    test = MNIST(args.data_dir, train=False, download=True, transform=image_transform())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=512, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, num_workers=0)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99,
                                   eps=1e-8, momentum=0, weight_decay=0)
    config = {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(out),
              "device": device, "dataset": "MNIST", "optimizer": "RMSprop",
              "train_size": len(train), "validation_size": len(val), "test_size": len(test),
              "parameters": sum(p.numel() for p in model.parameters()),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__),
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "split_indices.json").write_text(json.dumps({"train": train.indices, "val": val.indices}), encoding="utf-8")
    history, best_loss, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    print(json.dumps(config, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            count += targets.size(0)
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "train_accuracy": correct / count, "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"]}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "validation_loss": best_loss, "config": config}, out / "model.pt")
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    model, _ = load_model(out / "model.pt", device)
    # Тест используется только после завершения обучения и выбора эпохи.
    result = evaluate(model, test_loader, device)
    metrics = {k: v for k, v in result.items() if k not in ("labels", "predictions", "confidence")}
    metrics.update(best_epoch=best_epoch, validation_loss=best_loss,
                   elapsed_seconds=time.perf_counter() - started)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "actual", "predicted", "confidence"])
        writer.writerows((i, a, p, c) for i, (a, p, c) in enumerate(
            zip(result["labels"], result["predictions"], result["confidence"])))
    save_plots(history, result, test, out)
    examples = out.parent / "examples"
    examples.mkdir(exist_ok=True)
    from PIL import Image
    Image.fromarray(test.data[0].numpy()).save(examples / "mnist_test_0.png")
    print("TEST " + json.dumps(metrics), flush=True)
    return out


def prepare_image(path, mode="auto"):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("L")
    if mode == "mnist":
        if image.size != (28, 28):
            raise ValueError("В режиме mnist требуется изображение 28×28")
        return image
    array = np.asarray(image)
    if int(array.max()) - int(array.min()) < 5:
        raise ValueError("На изображении не найдена цифра")
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if mode == "dark" or (mode == "auto" and np.median(border) > 127):
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    array = np.asarray(image)
    ys, xs = np.where(array > 30)
    if len(xs) == 0:
        raise ValueError("На изображении не найдена цифра")
    image = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    width, height = image.size
    scale = 20 / max(width, height)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                         Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    array = np.asarray(canvas, dtype=float)
    y, x = np.indices(array.shape)
    dx = round(13.5 - float((x * array).sum() / array.sum()))
    dy = round(13.5 - float((y * array).sum() / array.sum()))
    # Не обрезаем штрихи, если центр масс сильно смещён к одному краю.
    left, top, right, bottom = canvas.getbbox()
    dx = min(max(dx, 1 - left), 27 - right)
    dy = min(max(dy, 1 - top), 27 - bottom)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(canvas, (dx, dy))
    return centered


def predict_main(argv=None):
    parser = argparse.ArgumentParser(description="Распознавание изображения одной цифры")
    parser.add_argument("--image", type=Path, help="Если не задано, откроется диалог выбора файла")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "results/model.pt")
    parser.add_argument("--output", type=Path, default=BASE / "results/prediction.png")
    parser.add_argument("--mode", choices=["auto", "dark", "light", "mnist"], default="auto",
                        help="dark: тёмная цифра; light: светлая; mnist: готовые 28×28")
    parser.add_argument("--show", action="store_true", help="Открыть результат в просмотрщике")
    args = parser.parse_args(argv)
    if args.image is None:
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
        root = tk.Tk()
        root.withdraw()
        path = askopenfilename(title="Выберите изображение одной цифры",
                               filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")])
        root.destroy()
        if not path:
            return
        args.image = Path(path)
    try:
        prepared = prepare_image(args.image, args.mode)
        model, checkpoint = load_model(args.checkpoint)
        with torch.inference_mode():
            probabilities = model(image_transform()(prepared).unsqueeze(0)).softmax(1)[0].numpy()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    prediction = int(probabilities.argmax())
    result = {"image": args.image.name, "mode": args.mode, "prediction": prediction,
              "confidence": float(probabilities[prediction]),
              "probabilities": probabilities.tolist(), "checkpoint_epoch": checkpoint["epoch"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    prepared.save(args.output.with_name(args.output.stem + "_input.png"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.7]})
    with Image.open(args.image) as original:
        axes[0].imshow(original.convert("RGB"))
    axes[0].set_title("Исходное изображение")
    axes[1].imshow(prepared, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Вход сети 28×28")
    for ax in axes[:2]:
        ax.axis("off")
    axes[2].bar(range(10), probabilities * 100)
    axes[2].set(xticks=range(10), xlabel="Цифра", ylabel="Softmax, %", ylim=(0, 105),
                title=f"Результат: {prediction}; p = {probabilities[prediction]:.2%}")
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show:
        with Image.open(args.output) as rendered:
            rendered.show()


def run_all(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-show", action="store_true")
    options, training_args = parser.parse_known_args(argv)
    out = train_main(training_args)
    examples = out.parent / "examples"
    # Отдельный рисунок служит примером внешнего файла, не частью MNIST.
    image = Image.new("RGB", (1120, 1120), "white")
    ImageDraw.Draw(image).line([(250, 280), (840, 280), (610, 590), (445, 900)],
                              fill="black", width=65, joint="curve")
    image.resize((280, 280), Image.Resampling.LANCZOS).save(examples / "digit_7.png")
    predict_main(["--image", str(examples / "digit_7.png"), "--checkpoint", str(out / "model.pt"),
                  "--output", str(out / "external_prediction.png")])
    predict_main(["--image", str(examples / "mnist_test_0.png"), "--mode", "mnist",
                  "--checkpoint", str(out / "model.pt"), "--output", str(out / "mnist_prediction.png")])
    print("\nРезультаты сохранены в", out.resolve())
    print("Сравнение с опубликованными результатами MNIST:")
    print("MCDNN (2012): 99.77%; ансамбль An et al. (2020): до 99.91%.")
    print("Это более сложные ансамбли с другими условиями обучения и отбора моделей.")
    print("Источники: https://arxiv.org/abs/1202.2745 ; https://arxiv.org/abs/2008.10400v2")
    if options.no_show:
        return
    show_results(out)


def show_results(out):
    """Показ всех графиков и выбор пользовательского изображения."""
    try:
        plt.switch_backend("TkAgg")
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
    except (ImportError, RuntimeError) as error:
        print("Не удалось открыть окна:", error)
        print("Графики сохранены в PNG. Для окон установите Python с Tcl/Tk.")
        return

    def display(path, title):
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.canvas.manager.set_window_title(title)
        ax.imshow(plt.imread(path))
        ax.axis("off")
        fig.tight_layout()
        plt.show(block=False)

    for filename, title in [
        ("learning_curves.png", "Потери и ошибка по эпохам"),
        ("confusion_matrix.png", "Матрица ошибок"),
        ("test_examples.png", "Примеры тестовой выборки"),
        ("test_errors.png", "Ошибки распознавания"),
        ("external_prediction.png", "Распознавание внешнего рисунка"),
        ("mnist_prediction.png", "Контрольное изображение MNIST"),
    ]:
        display(out / filename, title)
    plt.pause(0.2)
    root = plt.figure(plt.get_fignums()[0]).canvas.manager.window
    path = askopenfilename(parent=root, title="Выберите свою цифру или нажмите Отмена",
                           filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")])
    if path:
        try:
            predict_main(["--image", path, "--checkpoint", str(out / "model.pt"),
                          "--output", str(out / "user_prediction.png")])
            display(out / "user_prediction.png", "Результат для выбранного изображения")
        except SystemExit as error:
            print("Не удалось распознать выбранный файл:", error)
    print("Для завершения программы закройте окна графиков.")
    plt.show()


def test_main(argv=None):
    parser = argparse.ArgumentParser(description="Оценка сохранённых весов на 10 000 изображений MNIST")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "results/model.pt")
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, checkpoint = load_model(args.checkpoint, device)
    test = MNIST(args.data_dir, train=False, download=True, transform=image_transform())
    result = evaluate(model, DataLoader(test, batch_size=512), device)
    metrics = {k: v for k, v in result.items() if k not in ("labels", "predictions", "confidence")}
    metrics["checkpoint_epoch"] = checkpoint["epoch"]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) == 1:
        run_all([])
        return
    command = sys.argv[1]
    if command in ("-h", "--help"):
        print(__doc__)
        print("Справка по параметрам: python lab1.py train --help")
        print("                       python lab1.py predict --help")
        print("                       python lab1.py test --help")
        return
    actions = {"all": run_all, "train": train_main, "predict": predict_main, "test": test_main}
    if command not in actions:
        raise SystemExit("Допустимые команды: all, train, predict, test. Справка: --help")
    actions[command](sys.argv[2:])


if __name__ == "__main__":
    main()
