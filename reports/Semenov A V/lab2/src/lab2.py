
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
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


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
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "correct": correct,
        "total": count,
        "errors": count - correct,
        "labels": labels,
        "predictions": predictions,
        "confidence": probabilities,
    }


def save_plots(history, result, test, out):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for prefix, label in [("train", "Обучение"), ("val", "Валидация")]:
        axes[0].plot(epochs, [r[prefix + "_loss"] for r in history], "o-", label=label)
        axes[1].plot(
            epochs,
            [100 * (1 - r[prefix + "_accuracy"]) for r in history],
            "o-",
            label=label,
        )
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
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set(
        xticks=range(10),
        yticks=range(10),
        xlabel="Предсказанная цифра",
        ylabel="Истинная цифра",
    )
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    for name, indices, title in [
        ("test_examples", list(range(12)), "Первые 12 изображений тестовой выборки"),
        (
            "test_errors",
            [
                i
                for i, (a, p) in enumerate(zip(result["labels"], result["predictions"]))
                if a != p
            ][:12],
            "Первые ошибки в порядке тестовой выборки",
        ),
    ]:
        fig, axes = plt.subplots(2, 6, figsize=(10, 3.8), layout="constrained")
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, indices):
            ax.imshow(test.data[index], cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f'№{index}: {result["labels"][index]} → {result["predictions"][index]}\n'
                f'p = {100 * result["confidence"][index]:.1f}%',
                fontsize=10,
            )
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
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    full_train = MNIST(
        args.data_dir, train=True, download=True, transform=image_transform()
    )
    train, val = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed)
    )
    test = MNIST(args.data_dir, train=False, download=True, transform=image_transform())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val, batch_size=512, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, num_workers=0)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, alpha=0.99, eps=1e-8, momentum=0, weight_decay=0
    )
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(out),
        "device": device,
        "dataset": "MNIST",
        "optimizer": "RMSprop",
        "train_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
        "parameters": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "split_indices.json").write_text(
        json.dumps({"train": train.indices, "val": val.indices}), encoding="utf-8"
    )
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
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "config": config,
                },
                out / "model.pt",
            )
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    model, _ = load_model(out / "model.pt", device)
    # Тест используется только после завершения обучения и выбора эпохи.
    result = evaluate(model, test_loader, device)
    metrics = {
        k: v
        for k, v in result.items()
        if k not in ("labels", "predictions", "confidence")
    }
    metrics.update(
        best_epoch=best_epoch,
        validation_loss=best_loss,
        elapsed_seconds=time.perf_counter() - started,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "actual", "predicted", "confidence"])
        writer.writerows(
            (i, a, p, c)
            for i, (a, p, c) in enumerate(
                zip(result["labels"], result["predictions"], result["confidence"])
            )
        )
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
    image = image.crop(
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    )
    width, height = image.size
    scale = 20 / max(width, height)
    image = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
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


from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.nn import functional as F


class TransferMNIST(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.network = mobilenet_v3_large(weights=weights)
        self.network.classifier[-1] = nn.Linear(1280, 10)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
        x = x.expand(-1, 3, -1, -1)
        return self.network((x - self.mean) / self.std)

    def set_phase(self, fine_tune):
        for p in self.network.features.parameters():
            p.requires_grad_(False)
        if fine_tune:
            for p in self.network.features[13:].parameters():
                p.requires_grad_(True)

    def training_mode(self, fine_tune):
        self.train()
        # Статистики BatchNorm замороженной части тоже не обновляются.
        self.network.features.eval()
        if fine_tune:
            self.network.features[13:].train()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def train_mobile(data, out, device):
    out.mkdir(parents=True, exist_ok=True)
    random.seed(16)
    np.random.seed(16)
    torch.manual_seed(16)
    full = MNIST(data, train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(
        full, [55000, 5000], generator=torch.Generator().manual_seed(16)
    )
    test = MNIST(data, train=False, download=True, transform=transforms.ToTensor())
    loaders = [
        DataLoader(
            train,
            batch_size=96,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(16),
        ),
        DataLoader(val, batch_size=96),
        DataLoader(test, batch_size=96),
    ]
    save_json(out / "split_indices.json", {"train": train.indices, "val": val.indices})
    model = TransferMNIST().to(device)
    model.set_phase(False)
    optimizer = torch.optim.RMSprop(model.network.classifier.parameters(), lr=0.001)
    history, best_loss, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    config = {
        "architecture": "MobileNetV3 Large",
        "weights": "IMAGENET1K_V2",
        "parameters": sum(p.numel() for p in model.parameters()),
        "head_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "seed": 16,
        "batch_size": 96,
        "epochs": 8,
        "optimizer": "RMSprop",
        "device": device,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }
    for epoch in range(1, 9):
        fine_tune = epoch > 2
        if epoch == 3:
            model.set_phase(True)
            config["fine_tune_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            optimizer = torch.optim.RMSprop(
                [
                    {"params": model.network.features[13:].parameters(), "lr": 0.0001},
                    {"params": model.network.classifier.parameters(), "lr": 0.0003},
                ]
            )
        model.training_mode(fine_tune)
        loss_sum, correct, count = 0.0, 0, 0
        for images, targets in loaders[0]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            count += len(targets)
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
        validation = evaluate(model, loaders[1], device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_loss,
                },
                out / "model.pt",
            )
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print("MOBILE " + json.dumps(row), flush=True)
    model.load_state_dict(
        torch.load(out / "model.pt", map_location=device, weights_only=True)[
            "model_state"
        ]
    )
    result = evaluate(model, loaders[2], device)
    metrics = {
        k: v
        for k, v in result.items()
        if k not in ("labels", "predictions", "confidence")
    }
    metrics.update(
        best_epoch=best_epoch,
        validation_loss=best_loss,
        elapsed_seconds=time.perf_counter() - started,
    )
    save_json(out / "metrics.json", metrics)
    save_json(out / "config.json", config)
    save_json(out / "test_predictions.json", result)
    save_plots(history, result, test, out)
    print("MOBILE TEST " + json.dumps(metrics), flush=True)


def load_pair(out, device="cpu"):
    cnn, _ = load_model(out / "custom/model.pt", device)
    mobile = TransferMNIST(False).to(device)
    mobile.load_state_dict(
        torch.load(out / "mobile/model.pt", map_location=device, weights_only=True)[
            "model_state"
        ]
    )
    mobile.eval()
    return cnn, mobile


@torch.inference_mode()
def compare_image(path, out, mode="auto", name="user_prediction"):
    prepared = prepare_image(path, mode)
    cnn, mobile = load_pair(out)
    raw = transforms.ToTensor()(prepared).unsqueeze(0)
    probabilities = [
        cnn((raw - 0.5) / 0.5).softmax(1)[0].numpy(),
        mobile(raw).softmax(1)[0].numpy(),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), layout="constrained")
    with Image.open(path) as original:
        axes[0].imshow(original.convert("RGB"))
    axes[0].set_title("Исходное изображение")
    axes[1].imshow(prepared, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Подготовленная цифра 28×28")
    for ax in axes[:2]:
        ax.axis("off")
    result = {}
    for ax, p, label in zip(
        axes[2:], probabilities, ["Сеть из ЛР 1", "MobileNetV3 Large"]
    ):
        pred = int(p.argmax())
        ax.bar(range(10), p * 100)
        ax.set(
            xticks=range(10),
            ylim=(0, 105),
            xlabel="Цифра",
            ylabel="Softmax, %",
            title=f"{label}\nКласс {pred}; p = {p[pred]:.2%}",
        )
        result[label] = {"prediction": pred, "probabilities": p.tolist()}
    fig.savefig(out / (name + ".png"), dpi=180)
    plt.close(fig)
    save_json(out / (name + ".json"), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def comparison_plot(out):
    records = [
        json.loads((out / n / "metrics.json").read_text()) for n in ["custom", "mobile"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    labels = ["Сеть из ЛР 1", "MobileNetV3 Large"]
    for ax, key, title, factor in [
        (axes[0], "accuracy", "Точность на тесте, %", 100),
        (axes[1], "errors", "Ошибки на 10 000 изображений", 1),
    ]:
        bars = ax.bar(
            labels,
            [r[key] * factor for r in records],
            color=["steelblue", "darkorange"],
        )
        ax.bar_label(bars, fmt="%.2f" if factor == 100 else "%d")
        ax.set_title(title)
        ax.set_ylim(0, 110 if factor == 100 else max(r[key] for r in records) * 1.2)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


def show_results(out):
    try:
        plt.switch_backend("TkAgg")
        from tkinter.filedialog import askopenfilename

        def display(path):
            fig, ax = plt.subplots(figsize=(11, 6))
            fig.canvas.manager.set_window_title(str(path.relative_to(out)))
            ax.imshow(plt.imread(path))
            ax.axis("off")
            fig.tight_layout()
            plt.show(block=False)

        paths = [
            out / n / f
            for n in ["custom", "mobile"]
            for f in [
                "learning_curves.png",
                "confusion_matrix.png",
                "test_examples.png",
                "test_errors.png",
            ]
        ]
        paths += [
            out / "comparison.png",
            out / "external_prediction.png",
            out / "mnist_prediction.png",
        ]
        for path in paths:
            display(path)
        plt.pause(0.2)
        root = plt.figure(plt.get_fignums()[0]).canvas.manager.window
        path = askopenfilename(
            parent=root,
            title="Выберите изображение одной цифры или нажмите Отмена",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            try:
                compare_image(Path(path), out)
                display(out / "user_prediction.png")
            except (ValueError, OSError) as error:
                print("Ошибка изображения:", error)
        print("Для завершения закройте окна графиков.")
        plt.show()
    except (ImportError, RuntimeError) as error:
        print("Графики сохранены в PNG; окна недоступны:", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "lab2_data")
    parser.add_argument("--output-dir", type=Path, default=BASE / "lab2_results")
    parser.add_argument("--weights-dir", type=Path, default=BASE / "lab2_weights")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--predict",
        type=Path,
        help="Распознать файл сохранёнными моделями без обучения",
    )
    parser.add_argument(
        "--mode", choices=["auto", "dark", "light", "mnist"], default="auto"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.hub.set_dir(str(args.weights_dir))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.predict:
        compare_image(args.predict, out, args.mode)
        if not args.no_show:
            Image.open(out / "user_prediction.png").show()
        return
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    train_main(
        [
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(out / "custom"),
            "--device",
            device,
        ]
    )
    train_mobile(args.data_dir, out / "mobile", device)
    comparison_plot(out)
    examples = out / "examples"
    examples.mkdir(exist_ok=True)
    image = Image.new("RGB", (1120, 1120), "white")
    ImageDraw.Draw(image).line(
        [(250, 280), (840, 280), (610, 590), (445, 900)],
        fill="black",
        width=65,
        joint="curve",
    )
    image.resize((280, 280), Image.Resampling.LANCZOS).save(examples / "digit_7.png")
    compare_image(examples / "digit_7.png", out, name="external_prediction")
    compare_image(
        examples / "mnist_test_0.png", out, mode="mnist", name="mnist_prediction"
    )
    print("Результаты:", out.resolve())
    print(
        "Опубликованные результаты MNIST: MCDNN — 99.77%; ансамбль An et al. — до 99.91%."
    )
    print("Условия обучения и отбора моделей отличаются от данного эксперимента.")
    print("https://arxiv.org/abs/1202.2745 ; https://arxiv.org/abs/2008.10400v2")
    if not args.no_show:
        show_results(out)


if __name__ == "__main__":
    main()
