import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights

import numpy as np
import matplotlib.pyplot as plt


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"Используемое устройство: {device}")
    if device.type == 'cuda':
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    NUM_CLASSES = 10


    print("\n[1/4] ЗАГРУЗКА КАСТОМНОЙ МОДЕЛИ ИЗ ЛР №1...")

    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)

    transform_test_custom = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_set_custom = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                   transform=transform_test_custom)
    test_loader_custom = DataLoader(test_set_custom, batch_size=256, shuffle=False, num_workers=0)

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    custom_model = SimpleCNN().to(device)
    LAB1_PATH = 'cifar10_model.pth'

    if os.path.exists(LAB1_PATH):
        custom_model.load_state_dict(
            torch.load(LAB1_PATH, map_location=device, weights_only=True if hasattr(torch, 'serialization') else False))
        print(f"  Файл '{LAB1_PATH}' найден! Веса загружены.")
    else:
        print(f"  Файл '{LAB1_PATH}' не найден. Выполняется быстрое обучение...")
        transform_train_custom = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        train_set_custom = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=transform_train_custom)
        train_loader_custom = DataLoader(train_set_custom, batch_size=128, shuffle=True, num_workers=0)

        criterion_c = nn.CrossEntropyLoss()
        optimizer_c = optim.SGD(custom_model.parameters(), lr=0.01, momentum=0.9)
        custom_model.train()
        for ep in range(2):
            for imgs, lbls in train_loader_custom:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer_c.zero_grad()
                loss = criterion_c(custom_model(imgs), lbls)
                loss.backward()
                optimizer_c.step()

    custom_model.eval()
    correct_c, total_c = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader_custom:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = custom_model(imgs)
            correct_c += (outputs.argmax(1) == lbls).sum().item()
            total_c += lbls.size(0)

    custom_acc = 100.0 * correct_c / total_c
    print(f"  Точность SimpleCNN (ЛР1) на тесте: {custom_acc:.2f}%")


    print("\n[2/4] ОБУЧЕНИЕ ПРЕДОБУЧЕННОЙ МОДЕЛИ (ResNet34)...")

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform_train_pt = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test_pt = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set_pt = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_pt)
    test_set_pt = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_pt)

    train_loader_pt = DataLoader(train_set_pt, batch_size=128, shuffle=True, num_workers=0)
    test_loader_pt = DataLoader(test_set_pt, batch_size=256, shuffle=False, num_workers=0)

    weights = ResNet34_Weights.IMAGENET1K_V1
    pretrained_model = resnet34(weights=weights)

    for param in pretrained_model.parameters():
        param.requires_grad = False

    in_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(in_features, NUM_CLASSES)
    pretrained_model = pretrained_model.to(device)

    n_trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)

    criterion_pt = nn.CrossEntropyLoss()
    optimizer_pt = optim.SGD(pretrained_model.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    N_EPOCHS_PT = 3
    history_pt = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    def run_epoch_pt(loader, train=True):
        pretrained_model.train() if train else pretrained_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer_pt.zero_grad()
                outputs = pretrained_model(images)
                loss = criterion_pt(outputs, labels)
                if train:
                    loss.backward()
                    optimizer_pt.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    for epoch in range(1, N_EPOCHS_PT + 1):
        train_loss, train_acc = run_epoch_pt(train_loader_pt, train=True)
        test_loss, test_acc = run_epoch_pt(test_loader_pt, train=False)
        history_pt["train_loss"].append(train_loss)
        history_pt["train_acc"].append(train_acc)
        history_pt["test_loss"].append(test_loss)
        history_pt["test_acc"].append(test_acc)
        print(
            f"  ResNet34 | Эпоха {epoch:2d}/{N_EPOCHS_PT} | Train Acc: {train_acc * 100:5.2f}% | Test Acc: {test_acc * 100:5.2f}%")

    # График
    epochs_pt = range(1, N_EPOCHS_PT + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs_pt, history_pt["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(epochs_pt, history_pt["test_loss"], label="Test Loss", marker='o')
    axes[0].set_xlabel("Эпоха");
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ResNet34 — Ошибка (Loss)");
    axes[0].legend();
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_pt, [a * 100 for a in history_pt["train_acc"]], label="Train Acc", marker='o')
    axes[1].plot(epochs_pt, [a * 100 for a in history_pt["test_acc"]], label="Test Acc", marker='o')
    axes[1].set_xlabel("Эпоха");
    axes[1].set_ylabel("Accuracy, %")
    axes[1].set_title("ResNet34 — Точность (Accuracy)");
    axes[1].legend();
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


    print("\n[3/4] СРАВНЕНИЕ С SOTA НА CIFAR-10...")

    your_results = {
        "SimpleCNN (ЛР1)": {"acc": custom_acc, "params": 188_810, "note": "с нуля, ЛР1"},
        "ResNet34 (ЛР2)": {"acc": history_pt['test_acc'][-1] * 100, "params": n_trainable_params,
                           "note": "feature extraction, ЛР2"},
    }

    fair_baselines = {
        "VGG-16": {"acc": 92.6, "params": 15_000_000, "note": "с нуля"},
        "ResNet-18": {"acc": 93.0, "params": 11_000_000, "note": "с нуля"},
        "Wide ResNet-28-10": {"acc": 96.1, "params": 36_000_000, "note": "с аугментацией"},
    }

    sota_pretrained = {
        "ResNet-50": {"acc": 95.0, "params": 25_000_000, "note": "ImageNet pretrain"},
        "AutoAugment": {"acc": 98.5, "params": 26_000_000, "note": "PyramidNet + AA"},
        "ViT-H/14": {"acc": 99.5, "params": 632_000_000, "note": "JFT-300M pretrain"},
    }

    def print_table(title, data):
        print(f"\n--- {title} ---")
        print(f"{'Модель':<28}{'Точность':>10}{'Параметры':>15}   Примечание")
        print("-" * 75)
        for name, info in data.items():
            print(f"{name:<28}{info['acc']:>9.2f}%{info['params']:>15,}   {info['note']}")

    print_table("модели", your_results)
    print_table("Честные бейзлайны (обучение с нуля)", fair_baselines)
    print_table("SOTA с предобучением", sota_pretrained)

    all_models = {}
    all_models.update(your_results)
    all_models.update(fair_baselines)
    all_models.update(sota_pretrained)

    names = list(all_models.keys())
    accs = [all_models[n]['acc'] for n in names]
    colors = (
            ['crimson'] * len(your_results) +
            ['steelblue'] * len(fair_baselines) +
            ['seagreen'] * len(sota_pretrained)
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(names, accs, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Точность на CIFAR-10, %")
    ax.set_title("Сравнение моделей с бейзлайнами и SOTA (CIFAR-10)")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va='center', fontsize=9)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='crimson', label='Ваши модели (ЛР1 и ЛР2)'),
        Patch(facecolor='steelblue', label='Бейзлайны (с нуля)'),
        Patch(facecolor='seagreen', label='SOTA (с предобучением)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right')
    plt.tight_layout()
    plt.show()


    print("\n[4/4] ВИЗУАЛИЗАЦИЯ ДВОЙНОГО ПРЕДСКАЗАНИЯ ИЗОБРАЖЕНИЯ С РАБОЧЕГО СТОЛА...")

    # Возможные пути к файлу на Рабочем столе
    possible_paths = [
        r'C:\Users\Admin\Desktop\test.jpg',
        r'C:\Users\Admin\Desktop\test.png',
        r'C:\Users\Admin\Desktop\test.jpeg',
    ]

    img_path = None
    for p in possible_paths:
        if os.path.exists(p):
            img_path = p
            break

    if img_path and os.path.exists(img_path):
        print(f"  Загружаем изображение с Рабочего стола: {img_path}")
        img = Image.open(img_path).convert('RGB')
    else:
        print("  Картинка 'test.jpg' или 'test.png' не найдена на Рабочем столе.")
        print("  Берем случайную картинку из датасета CIFAR-10...")
        idx = random.randint(0, len(test_set_custom) - 1)
        tensor, label = test_set_custom[idx]
        img_np = tensor.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(CIFAR_STD) + np.array(CIFAR_MEAN)
        img_np = np.clip(img_np, 0, 1)
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        print(f"  Истинный класс выбранного изображения: {CLASSES[label]}")

    preprocess_custom = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    preprocess_pt = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    custom_model.eval()
    pretrained_model.eval()

    with torch.no_grad():
        probs_custom = F.softmax(custom_model(preprocess_custom(img).unsqueeze(0).to(device)),
                                 dim=1).cpu().numpy().flatten()
        probs_pt = F.softmax(pretrained_model(preprocess_pt(img).unsqueeze(0).to(device)),
                             dim=1).cpu().numpy().flatten()

    pred_custom = int(np.argmax(probs_custom))
    pred_pt = int(np.argmax(probs_pt))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title("Входное изображение")


    axes[1].barh(CLASSES, probs_custom * 100,
                 color=['red' if i == pred_custom else 'steelblue' for i in range(NUM_CLASSES)])
    axes[1].set_xlim(0, 100);
    axes[1].invert_yaxis()
    axes[1].set_title(f"SimpleCNN (ЛР1): {CLASSES[pred_custom]} ({probs_custom[pred_custom] * 100:.1f}%)")
    axes[1].set_xlabel("Вероятность, %")


    axes[2].barh(CLASSES, probs_pt * 100, color=['green' if i == pred_pt else 'steelblue' for i in range(NUM_CLASSES)])
    axes[2].set_xlim(0, 100);
    axes[2].invert_yaxis()
    axes[2].set_title(f"ResNet34 (ЛР2): {CLASSES[pred_pt]} ({probs_pt[pred_pt] * 100:.1f}%)")
    axes[2].set_xlabel("Вероятность, %")

    plt.tight_layout()
    plt.savefig('dual_inference_desktop.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()