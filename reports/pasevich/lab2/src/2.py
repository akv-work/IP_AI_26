import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from collections import defaultdict

from torchvision.models import densenet121, DenseNet121_Weights


def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    CLASSES = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
               'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm')

    NUM_CLASSES = 100

    MEAN = (0.5071, 0.4865, 0.4409)
    STD  = (0.2673, 0.2564, 0.2762)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True, transform=transform_test)

    BATCH_SIZE = 128
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    print(f"Train: {len(train_set)}, Test: {len(test_set)}, размер изображения: {train_set[0][0].shape}")

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, num_classes)
            self.dropout = nn.Dropout(0.4)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Обучаемых параметров: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    N_EPOCHS = 8

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        test_loss, test_acc = run_epoch(test_loader, train=False)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"Эпоха {epoch:2d}/{N_EPOCHS} | train loss {train_loss:.4f} acc {train_acc*100:5.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc*100:5.2f}%")

    custom_model = model

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    transform_train_pt = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test_pt = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set_pt = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                  download=True, transform=transform_train_pt)
    test_set_pt = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform_test_pt)

    BATCH_SIZE_PT = 64
    train_loader_pt = DataLoader(train_set_pt, batch_size=BATCH_SIZE_PT, shuffle=True, num_workers=2)
    test_loader_pt = DataLoader(test_set_pt, batch_size=128, shuffle=False, num_workers=2)

    print(f"Train: {len(train_set_pt)}, Test: {len(test_set_pt)}, размер изображения: {train_set_pt[0][0].shape}")

    weights = DenseNet121_Weights.IMAGENET1K_V1
    pretrained_model = densenet121(weights=weights)

    for param in pretrained_model.features.parameters():
        param.requires_grad = False

    in_features = pretrained_model.classifier.in_features
    pretrained_model.classifier = nn.Linear(in_features, NUM_CLASSES)

    pretrained_model = pretrained_model.to(device)

    n_params_total = sum(p.numel() for p in pretrained_model.parameters())
    n_params_trainable = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print(pretrained_model.classifier)
    print(f"Всего параметров: {n_params_total:,}")
    print(f"Обучаемых параметров (только классификатор): {n_params_trainable:,}")

    criterion_pt = nn.CrossEntropyLoss()
    optimizer_pt = optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=1e-3)

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
        print(f"Эпоха {epoch:2d}/{N_EPOCHS_PT} | train loss {train_loss:.4f} acc {train_acc*100:5.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc*100:5.2f}%")

    epochs_range_pt = range(1, N_EPOCHS_PT + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs_range_pt, history_pt["train_loss"], label="Train loss", marker='o', markersize=3)
    axes[0].plot(epochs_range_pt, history_pt["test_loss"], label="Test loss", marker='o', markersize=3)
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
    axes[0].set_title("DenseNet121: изменение ошибки"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range_pt, [a*100 for a in history_pt["train_acc"]], label="Train acc", marker='o', markersize=3)
    axes[1].plot(epochs_range_pt, [a*100 for a in history_pt["test_acc"]], label="Test acc", marker='o', markersize=3)
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("Accuracy, %")
    axes[1].set_title("DenseNet121: изменение точности"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Итоговая точность DenseNet121 на тесте: {history_pt['test_acc'][-1]*100:.2f}%")

    print("Сравнение результатов:")
    print(f"Кастомная СНС: {history['test_acc'][-1]*100:.2f}%  ({n_params:,} параметров, все обучаемые)")
    print(f"DenseNet121:   {history_pt['test_acc'][-1]*100:.2f}%  ({n_params_trainable:,} обучаемых параметров из {n_params_total:,})")

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ С SOTA НА CIFAR-100")
    print("=" * 70)

    your_results = {
        "SimpleCNN (ваша)":   {"acc": history['test_acc'][-1] * 100,    "params": n_params,           "note": "с нуля, 8 эпох"},
        "DenseNet121 (ваша)": {"acc": history_pt['test_acc'][-1] * 100, "params": n_params_trainable, "note": "только классификатор, 3 эпохи"},
    }

    fair_baselines = {
        "VGG-16":            {"acc": 72.7, "params": 20_000_000,  "note": "с нуля"},
        "ResNet-18":         {"acc": 75.6, "params": 11_000_000,  "note": "с нуля"},
        "ResNet-56":         {"acc": 76.5, "params": 24_000_000,  "note": "с нуля"},
        "Wide ResNet-28-10": {"acc": 81.2, "params": 36_000_000,  "note": "с аугментацией"},
    }

    sota_pretrained = {
        "DenseNet-BC (k=24)": {"acc": 82.8, "params":  27_000_000, "note": "ImageNet pretrain"},
        "EfficientNet-B7":    {"acc": 87.3, "params":  66_000_000, "note": "ImageNet pretrain"},
        "FixResNeXt-101":     {"acc": 89.9, "params":  84_000_000, "note": "ImageNet pretrain"},
        "ViT-L/16":           {"acc": 93.0, "params": 307_000_000, "note": "JFT-300M pretrain"},
        "ViT-H/14":           {"acc": 93.8, "params": 632_000_000, "note": "JFT-300M pretrain"},
    }

    def print_table(title, data):
        print(f"\n--- {title} ---")
        print(f"{'Модель':<28}{'Точность':>10}{'Параметры':>15}   Примечание")
        print("-" * 75)
        for name, info in data.items():
            print(f"{name:<28}{info['acc']:>9.2f}%{info['params']:>15,}   {info['note']}")

    print_table("Ваши модели", your_results)
    print_table("Честные бейзлайны (обучение с нуля)", fair_baselines)
    print_table("SOTA с предобучением (нельзя сравнивать напрямую)", sota_pretrained)

    all_models = {}
    all_models.update(your_results)
    all_models.update(fair_baselines)
    all_models.update(sota_pretrained)

    names = list(all_models.keys())
    accs  = [all_models[n]['acc'] for n in names]
    colors = (
        ['crimson'] * len(your_results) +
        ['steelblue'] * len(fair_baselines) +
        ['seagreen'] * len(sota_pretrained)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, accs, color=colors)
    ax.set_xlim(0, 102)
    ax.set_xlabel("Точность на CIFAR-100, %")
    ax.set_title("Сравнение ваших моделей с бейзлайнами и SOTA (CIFAR-100)")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                f"{acc:.2f}%", va='center', fontsize=9)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='crimson',   label='Ваши модели'),
        Patch(facecolor='steelblue', label='Бейзлайны (с нуля)'),
        Patch(facecolor='seagreen',  label='SOTA (с предобучением)'),
    ]
    ax.legend(handles=legend_elems, loc='center right')
    plt.tight_layout()
    plt.show()

    your_best = max(your_results.values(), key=lambda x: x['acc'])['acc']
    fair_best = max(fair_baselines.values(), key=lambda x: x['acc'])['acc']
    sota_best = max(sota_pretrained.values(), key=lambda x: x['acc'])['acc']

    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print(f"Лучшая ваша модель:              {your_best:.2f}%")
    print(f"Лучший честный бейзлайн:         {fair_best:.2f}%  (отставание: {fair_best - your_best:+.2f} п.п.)")
    print(f"Лучший SOTA (с предобучением):   {sota_best:.2f}%  (отставание: {sota_best - your_best:+.2f} п.п.)")
    print("\nПримечание: SOTA-модели используют предобучение на")
    print("ImageNet/JFT-300M, поэтому их сравнение с моделями,")
    print("обученными с нуля, не является строгим.")

    def predict_with_both(path):
        img = Image.open(path).convert('RGB')

        preprocess_custom = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        preprocess_pt = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        custom_model.eval()
        pretrained_model.eval()

        with torch.no_grad():
            probs_custom = F.softmax(custom_model(preprocess_custom(img).unsqueeze(0).to(device)), dim=1).cpu().numpy().flatten()
            probs_pt = F.softmax(pretrained_model(preprocess_pt(img).unsqueeze(0).to(device)), dim=1).cpu().numpy().flatten()

        pred_custom = int(np.argmax(probs_custom))
        pred_pt = int(np.argmax(probs_pt))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
        axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title("Входное изображение")

        axes[1].barh(CLASSES, probs_custom * 100, color=['red' if i == pred_custom else 'steelblue' for i in range(NUM_CLASSES)])
        axes[1].set_xlim(0, 100); axes[1].invert_yaxis()
        axes[1].set_title(f"Кастомная СНС: {CLASSES[pred_custom]}")
        axes[1].set_xlabel("Вероятность, %")

        axes[2].barh(CLASSES, probs_pt * 100, color=['green' if i == pred_pt else 'steelblue' for i in range(NUM_CLASSES)])
        axes[2].set_xlim(0, 100); axes[2].invert_yaxis()
        axes[2].set_title(f"DenseNet121: {CLASSES[pred_pt]}")
        axes[2].set_xlabel("Вероятность, %")

        plt.tight_layout()
        plt.show()
        return pred_custom, pred_pt

    predict_with_both(r'C:\Users\admin\Desktop\test1.png')


if __name__ == '__main__':
    main()