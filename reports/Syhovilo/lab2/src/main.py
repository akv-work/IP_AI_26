"""ОИИС, ЛР №2, вариант 18: CIFAR-10, RMSprop, SqueezeNet 1.1.

По умолчанию обучает предобученную SqueezeNet 1.1 и CNN из ЛР1 (если нет
локальных весов), сравнивает их на CIFAR-10 и сохраняет графики и прогнозы.

Установка: pip install -r requirements.txt
Запуск:     python main.py
Проверка:   python main.py --smoke-test
Фото:       python main.py --mode predict --image path/to/photo.jpg

Датасет и веса ImageNet скачиваются при первом полном запуске. Обучение
на CPU может занимать много времени. Демо-результаты не подставляются.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTPUT = BASE / "results"
WEIGHTS = BASE / "checkpoints"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LR1_ACCURACY = 75.23  # Реальный результат отчёта ЛР1; не результат нового запуска.


class CIFAR10CNN(nn.Module):
    """Точная архитектура из ЛР №1 Суховило В. С. (620 362 параметра)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(self.dropout(x))


def make_squeezenet(pretrained=True, fine_tune=False):
    """SqueezeNet 1.1, последние 1000 классов ImageNet -> 10 классов CIFAR."""
    weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
    model = models.squeezenet1_1(weights=weights)
    model.classifier[1] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    nn.init.normal_(model.classifier[1].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.classifier[1].bias)
    model.num_classes = len(CLASSES)
    if not fine_tune:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def preprocessing(image_size):
    """Две разные обработки для двух разных моделей."""
    baseline_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    baseline_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    pt_train = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    pt_test = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return baseline_train, baseline_test, pt_train, pt_test


def loaders(train_transform, test_transform, batch_size, device):
    train_data = datasets.CIFAR10(root=DATA, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root=DATA, train=False, download=True, transform=test_transform)
    options = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **options)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **options)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (output.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return {"loss": total_loss / total, "accuracy": 100 * correct / total}


def save_history(history, model_name):
    path = OUTPUT / f"{model_name}_history.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r["epoch"] for r in history]
    for axis, suffix, title in ((axes[0], "loss", "Ошибка"), (axes[1], "accuracy", "Точность")):
        axis.plot(epochs, [r[f"train_{suffix}"] for r in history], "o-", label="Обучение")
        axis.plot(epochs, [r[f"test_{suffix}"] for r in history], "o-", label="Тест")
        axis.set_title(f"{model_name}: {title}")
        axis.set_xlabel("Эпоха")
        axis.set_ylabel("%" if suffix == "accuracy" else "CrossEntropyLoss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{model_name}_curves.png", dpi=160)
    plt.close(fig)


def train_or_load(model, name, train_loader, test_loader, device, epochs, lr):
    path = WEIGHTS / f"{name}.pth"
    model = model.to(device)
    if path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[{name}] Загружены сохранённые веса: {path}")
        return model, checkpoint.get("metrics")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test = run_epoch(model, test_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train["loss"],
                        "train_accuracy": train["accuracy"], "test_loss": test["loss"],
                        "test_accuracy": test["accuracy"]})
        print(f"[{name}] Эпоха {epoch}/{epochs}: "
              f"train loss={train['loss']:.4f}, acc={train['accuracy']:.2f}%; "
              f"test loss={test['loss']:.4f}, acc={test['accuracy']:.2f}%", flush=True)

    save_history(history, name)
    metrics = history[-1]
    torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                "epochs": epochs, "lr": lr}, path)
    print(f"[{name}] Веса сохранены: {path}")
    return model, metrics


def predict_both(image, baseline, pretrained, baseline_transform, pt_transform, device, target=None, filename="comparison.png"):
    image = image.convert("RGB")
    predictions = []
    for model, transform in ((baseline, baseline_transform), (pretrained, pt_transform)):
        model.eval()
        with torch.no_grad():
            tensor = transform(image).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        predictions.append(probabilities)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1.5, 1.5]})
    axes[0].imshow(image)
    axes[0].set_title("Изображение" + (f"\nИстинный класс: {CLASSES[target]}" if target is not None else ""))
    axes[0].axis("off")
    for axis, values, title in zip(axes[1:], predictions, ("CNN из ЛР1", "SqueezeNet 1.1 из ЛР2")):
        label = int(np.argmax(values))
        bars = ["#366a94"] * len(CLASSES)
        bars[label] = "#24955a"
        axis.barh(CLASSES, values * 100, color=bars)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Вероятность, %")
        axis.set_title(f"{title}\n{CLASSES[label]}: {values[label]*100:.2f}%")
    fig.tight_layout()
    destination = OUTPUT / filename
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(f"Визуализация: {destination}")
    return [int(np.argmax(probs)) for probs in predictions]


def compare_accuracy(baseline, pretrained, test_loader_baseline, test_loader_pt, device):
    loss_fn = nn.CrossEntropyLoss()
    old = run_epoch(baseline, test_loader_baseline, loss_fn, device)
    new = run_epoch(pretrained, test_loader_pt, loss_fn, device)
    print(f"\nПОВТОРНО ИЗМЕРЕНО: CNN = {old['accuracy']:.2f}%, "
          f"SqueezeNet 1.1 = {new['accuracy']:.2f}%")
    print(f"Опубликовано в отчёте ЛР1: CNN = {LR1_ACCURACY:.2f}% (другой запуск)")
    summary = {"variant": 18, "dataset": "CIFAR-10", "optimizer": "RMSprop",
               "baseline_reported_lr1_accuracy": LR1_ACCURACY,
               "cnn_accuracy_remeasured": old["accuracy"],
               "squeezenet_accuracy_remeasured": new["accuracy"],
               "difference_percentage_points": round(new["accuracy"] - old["accuracy"], 4)}
    (OUTPUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def smoke_test(device, image_size):
    cnn = CIFAR10CNN().to(device).eval()
    snet = make_squeezenet(pretrained=False).to(device).eval()
    with torch.no_grad():
        a = cnn(torch.zeros(2, 3, 32, 32, device=device))
        b = snet(torch.zeros(2, 3, image_size, image_size, device=device))
    assert a.shape == b.shape == (2, 10)
    assert sum(p.numel() for p in cnn.parameters()) == 620362
    assert isinstance(snet.classifier[1], nn.Conv2d) and snet.classifier[1].out_channels == 10
    print("SMOKE TEST OK: обе модели принимают изображения и выдают 10 классов")


def main():
    parser = argparse.ArgumentParser(description="ОИИС, ЛР2, вариант 18")
    parser.add_argument("--mode", choices=["full", "predict"], default="full")
    parser.add_argument("--image", type=Path, help="Произвольная фотография для классификации")
    parser.add_argument("--epochs-pretrained", type=int, default=8)
    parser.add_argument("--epochs-baseline", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fine-tune", action="store_true", help="Обучать и backbone SqueezeNet, а не только классификатор")
    parser.add_argument("--smoke-test", action="store_true", help="Только проверить архитектуры без загрузки данных")
    args = parser.parse_args()
    if args.image_size < 32 or args.epochs_pretrained < 1 or args.epochs_baseline < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Проверьте размер изображений, число эпох, размер батча и learning rate")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if args.smoke_test:
        smoke_test(device, args.image_size)
        return

    for directory in (DATA, OUTPUT, WEIGHTS):
        directory.mkdir(parents=True, exist_ok=True)
    base_train, base_test, pre_train, pre_test = preprocessing(args.image_size)
    if args.mode == "predict":
        base_model = CIFAR10CNN().to(device)
        pre_model = make_squeezenet(pretrained=False).to(device)
        for name, model in (("cnn_lr1", base_model), ("squeezenet11_lr2", pre_model)):
            path = WEIGHTS / f"{name}.pth"
            if not path.exists():
                parser.error(f"Нет весов {path}. Сначала выполните python main.py")
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
    else:
        train_base, test_base = loaders(base_train, base_test, args.batch_size, device)
        train_pre, test_pre = loaders(pre_train, pre_test, args.batch_size, device)
        base_model, _ = train_or_load(CIFAR10CNN(), "cnn_lr1", train_base, test_base,
                                      device, args.epochs_baseline, args.lr)
        pre_path = WEIGHTS / "squeezenet11_lr2.pth"
        pre_model = make_squeezenet(pretrained=not pre_path.exists(), fine_tune=args.fine_tune)
        pre_model, _ = train_or_load(pre_model, "squeezenet11_lr2", train_pre, test_pre,
                                     device, args.epochs_pretrained, args.lr)
        summary = compare_accuracy(base_model, pre_model, test_base, test_pre, device)
        test_images = datasets.CIFAR10(root=DATA, train=False, download=False)
        for index in (1, 234, 1789):
            photo = Image.fromarray(test_images.data[index])
            predict_both(photo, base_model, pre_model, base_test, pre_test, device,
                         target=int(test_images.targets[index]), filename=f"example_{index}.png")
        print("\nРезультаты для отчёта: " + str(OUTPUT / "comparison.json"))
        print("Разница с результатом старого отчёта ЛР1 не выдаётся за парный эксперимент.")

    if args.image:
        if not args.image.is_file():
            parser.error(f"Изображение не найдено: {args.image}")
        with Image.open(args.image) as img:
            predict_both(img, base_model, pre_model, base_test, pre_test, device,
                         filename="custom_image.png")


if __name__ == "__main__":
    main()
