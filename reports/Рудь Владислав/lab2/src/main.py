"""
Лабораторная работа №2
Вариант 15: STL-10, Adadelta, MobileNetV2 (pre-trained)

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from io import BytesIO
import torch.nn.functional as F
from urllib.request import urlopen
import os


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("⚠️  GPU не найдена, обучение на CPU")


batch_size = 64
num_epochs = 25
lr = 1.0
num_classes = 10
num_workers = 4

stl10_classes = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
STL10_MEAN = [0.4469, 0.4399, 0.4066]
STL10_STD  = [0.2603, 0.2566, 0.2713]



class SimpleSTL10CNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleSTL10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_sum / total, 100 * correct / total



def load_image_from_url(url):
    with urlopen(url, timeout=15) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def denorm(img_tensor, mean, std):
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = np.array(std) * img + np.array(mean)
    return np.clip(img, 0, 1)


def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return stl10_classes[pred.item()], conf.item() * 100



def main():
    # ---------- ДАННЫЕ ----------
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = torchvision.datasets.STL10(
        root="./data", split="train", download=True, transform=train_transform)
    test_dataset = torchvision.datasets.STL10(
        root="./data", split="test", download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"\nОбучающая выборка: {len(train_dataset)}")
    print(f"Тестовая выборка:  {len(test_dataset)}\n")


    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Обучаемых параметров: {trainable:,}")
    print(f"Всего параметров:     {total:,}\n")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.classifier.parameters(), lr=lr)

    # Mixed Precision (только на GPU)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    train_losses, test_losses, test_accs = [], [], []
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)

        mem_info = ""
        if device.type == "cuda":
            mem_info = f" | GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Time: {time.time()-epoch_start:.1f}s{mem_info}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ Обучение завершено за {elapsed:.1f} мин")
    print(f"🏆 Лучшая точность на тесте: {best_acc:.2f}%")

    torch.save(model.state_dict(), "mobilenetv2_stl10.pth")
    print("💾 Веса сохранены в mobilenetv2_stl10.pth")


    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(test_losses, label="Test Loss", linewidth=2)
    plt.xlabel("Эпоха"); plt.ylabel("Loss"); plt.legend()
    plt.title("Изменение ошибки"); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label="Test Accuracy", color="green", linewidth=2)
    plt.xlabel("Эпоха"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.title("Точность на тестовой выборке"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mobilenetv2_stl10_training.png", dpi=150)
    plt.show()


    custom_model = None
    if os.path.exists("custom_stl10.pth"):
        custom_model = SimpleSTL10CNN().to(device)
        custom_model.load_state_dict(torch.load("custom_stl10.pth", map_location=device))
        custom_model.eval()
        print("\n✅ Кастомная модель загружена для сравнения")
    else:
        print("\n⚠️  custom_stl10.pth не найден — сравнение будет только с теорией")
        print("   Запустите ЛР №1 и сохраните веса командой:")
        print("   torch.save(model.state_dict(), 'custom_stl10.pth')")


    if custom_model is not None:
        custom_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
        ])
        custom_test_dataset = torchvision.datasets.STL10(
            root="./data", split="test", download=False,
            transform=custom_test_transform
        )
        custom_test_loader = DataLoader(
            custom_test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        _, custom_acc = evaluate(custom_model, custom_test_loader, criterion)

        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ МОДЕЛЕЙ (ЛР №1 vs ЛР №2)")
        print("=" * 70)
        print(f"{'Метрика':<28}{'SimpleSTL10CNN':<22}{'MobileNetV2':<20}")
        print("-" * 70)
        custom_params = sum(p.numel() for p in custom_model.parameters())
        print(f"{'Всего параметров':<28}{custom_params:<22,}{total:<20,}")
        print(f"{'Обучаемых параметров':<28}{custom_params:<22,}{trainable:<20,}")
        print(f"{'Оптимизатор':<28}{'Adadelta (lr=1.0)':<22}{'Adadelta (lr=1.0)':<20}")
        print(f"{'Test Accuracy (%)':<28}{custom_acc:<22.2f}{best_acc:<20.2f}")
        print("=" * 70)


    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ НА ПРОИЗВОЛЬНЫХ ИЗОБРАЖЕНИЯХ ИЗ ИНТЕРНЕТА")
    print("=" * 70)

    viz_mobilenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    viz_custom = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)
    ])


    test_urls = [
        ("dog", "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400"),
        ("cat", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"),
        ("ship", "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=400"),
        ("airplane", "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400"),
        ("car", "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400"),
    ]

    for true_cls, url in test_urls:
        try:
            pil_img = load_image_from_url(url)
        except Exception as e:
            print(f"⚠️  Не удалось загрузить {url}: {e}")
            continue

        t_mobile = viz_mobilenet(pil_img)
        t_custom = viz_custom(pil_img)

        pred_mb, conf_mb = predict(model, t_mobile)

        n_models = 2 if custom_model is not None else 1
        fig, axes = plt.subplots(1, 2 + (1 if custom_model else 0),
                                 figsize=(5 * (2 + (1 if custom_model else 0)), 5))

        axes[0].imshow(pil_img)
        axes[0].set_title(f"Оригинал (истина: {true_cls})")
        axes[0].axis("off")

        axes[1].imshow(denorm(t_mobile, IMAGENET_MEAN, IMAGENET_STD))
        c = "green" if pred_mb == true_cls else "red"
        axes[1].set_title(f"MobileNetV2 (ЛР №2)\n{pred_mb} ({conf_mb:.1f}%)", color=c)
        axes[1].axis("off")

        if custom_model is not None:
            pred_cu, conf_cu = predict(custom_model, t_custom)
            axes[2].imshow(denorm(t_custom, STL10_MEAN, STL10_STD))
            c = "green" if pred_cu == true_cls else "red"
            axes[2].set_title(f"SimpleSTL10CNN (ЛР №1)\n{pred_cu} ({conf_cu:.1f}%)",
                              color=c)
            axes[2].axis("off")
            print(f"{true_cls:<10} | MobileNetV2: {pred_mb:<10} ({conf_mb:5.1f}%) | "
                  f"Custom: {pred_cu:<10} ({conf_cu:5.1f}%)")
        else:
            print(f"{true_cls:<10} | MobileNetV2: {pred_mb:<10} ({conf_mb:5.1f}%)")

        plt.tight_layout()
        plt.savefig(f"compare_{true_cls}.png", dpi=120)
        plt.show()


if __name__ == "__main__":
    main()