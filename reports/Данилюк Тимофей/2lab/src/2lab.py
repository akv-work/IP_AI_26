import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --- 1. Настройка устройства ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# --- 2. Трансформации ---
transform_train_resnet = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test_resnet = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train_custom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
transform_test_custom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# --- 3. Загрузка датасета ---
train_dataset_resnet = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train_resnet)
test_dataset_resnet = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test_resnet)
train_loader_resnet = torch.utils.data.DataLoader(train_dataset_resnet, batch_size=64, shuffle=True)
test_loader_resnet = torch.utils.data.DataLoader(test_dataset_resnet, batch_size=64, shuffle=False)

train_dataset_custom = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train_custom)
test_dataset_custom = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test_custom)
train_loader_custom = torch.utils.data.DataLoader(train_dataset_custom, batch_size=64, shuffle=True)
test_loader_custom = torch.utils.data.DataLoader(test_dataset_custom, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()

# ============================================================
# ЧАСТЬ 1: ОБУЧЕНИЕ ПРЕДОБУЧЕННОГО RESNET18
# ============================================================
print("\n--- ЧАСТЬ 1: Начало обучения ResNet18 (Fashion-MNIST, SGD) ---")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)
resnet18 = resnet18.to(device)

optimizer_resnet = optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9)
epochs_resnet = 5

train_losses_resnet = []
test_accuracies_resnet = []

for epoch in range(epochs_resnet):
    resnet18.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader_resnet):
        images, labels = images.to(device), labels.to(device)
        optimizer_resnet.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_resnet.step()
        running_loss += loss.item() * images.size(0)

        if (i + 1) % 200 == 0:
            print(
                f"ResNet18 | Эпоха [{epoch + 1}/{epochs_resnet}], Шаг [{i + 1}/{len(train_loader_resnet)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader_resnet.dataset)
    train_losses_resnet.append(epoch_loss)

    resnet18.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader_resnet:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies_resnet.append(accuracy)
    print(
        f"--- ResNet18 Итог Эпохи [{epoch + 1}/{epochs_resnet}]: Loss = {epoch_loss:.4f} | Test Accuracy = {accuracy:.2f}% ---")

# --- ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ RESNET18 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_resnet + 1), train_losses_resnet, marker='o', color='blue')
plt.title('ResNet18: График изменения ошибки (Loss)')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_resnet + 1), test_accuracies_resnet, marker='o', color='green')
plt.title('ResNet18: График точности (Accuracy)')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.tight_layout()
plt.show()  # <--- Окно с графиками появится здесь!

# ============================================================
# ЧАСТЬ 2: ОБУЧЕНИЕ КАСТОМНОЙ СЕТИ ИЗ ЛР №1
# ============================================================
print("\n--- ЧАСТЬ 2: Начало обучения Кастомной CNN (ЛР №1) ---")


class SimpleCNN_LR2_v2(nn.Module):
    def __init__(self):
        super(SimpleCNN_LR2_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


custom_model = SimpleCNN_LR2_v2().to(device)
optimizer_custom = optim.SGD(custom_model.parameters(), lr=0.01, momentum=0.9)
epochs_custom = 3

for epoch in range(epochs_custom):
    custom_model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader_custom):
        images, labels = images.to(device), labels.to(device)
        optimizer_custom.zero_grad()
        outputs = custom_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_custom.step()
        running_loss += loss.item() * images.size(0)

        if (i + 1) % 200 == 0:
            print(
                f"Custom CNN | Эпоха [{epoch + 1}/{epochs_custom}], Шаг [{i + 1}/{len(train_loader_custom)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader_custom.dataset)

    custom_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader_custom:
            images, labels = images.to(device), labels.to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(
        f"--- Custom CNN Итог Эпохи [{epoch + 1}/{epochs_custom}]: Loss = {epoch_loss:.4f} | Test Accuracy = {accuracy:.2f}% ---")

torch.save(custom_model.state_dict(), 'custom_cnn_lr1.pt')
print("✓ Кастомная модель обучена и сохранена в 'custom_cnn_lr1.pt'")

# ============================================================
# ЧАСТЬ 3: ГЕНЕРАЦИЯ КАРТИНКИ И ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n--- ЧАСТЬ 3: Генерация тестовой картинки и сравнение ---")
TARGET_CLASS = 3
TARGET_FILENAME = 'item.png'

found_idx = None
for i in range(len(test_dataset_custom)):
    _, sample_label = test_dataset_custom[i]
    if sample_label == TARGET_CLASS:
        found_idx = i
        break

if found_idx is not None:
    sample_img, sample_label = test_dataset_custom[found_idx]
    img_denorm = sample_img * 0.3530 + 0.2860
    img_denorm = torch.clamp(img_denorm, 0, 1)
    pil_img = to_pil_image(img_denorm[0])
    pil_img.save(TARGET_FILENAME)
    print(f"✓ Сохранили {TARGET_FILENAME} (Истинный класс: {FASHION_CLASSES[sample_label]})")

img = Image.open(TARGET_FILENAME).convert('L')

# 1. ResNet18
img_resnet = transform_test_resnet(img).unsqueeze(0).to(device)
resnet18.eval()
with torch.no_grad():
    out_resnet = resnet18(img_resnet)
    _, pred_resnet = torch.max(out_resnet, 1)
pred_resnet_class = FASHION_CLASSES[pred_resnet.item()]
print(f"ResNet18 предсказал: {pred_resnet_class}")

# 2. Custom CNN
img_custom = transform_test_custom(img).unsqueeze(0).to(device)
custom_model.eval()
with torch.no_grad():
    out_custom = custom_model(img_custom)
    _, pred_custom = torch.max(out_custom, 1)
pred_custom_class = FASHION_CLASSES[pred_custom.item()]
print(f"Custom CNN (ЛР1) предсказал: {pred_custom_class}")

# 3. Отрисовка графиков сравнения
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Оригинал', fontsize=10)
axes[0].axis('off')

axes[1].imshow(img, cmap='gray')
axes[1].set_title(f'ResNet18\n{pred_resnet_class}', fontsize=10, color='blue', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(img, cmap='gray')
axes[2].set_title(f'Custom CNN (ЛР1)\n{pred_custom_class}', fontsize=10, color='green', fontweight='bold')
axes[2].axis('off')

plt.suptitle('Сравнение моделей (Fashion-MNIST)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

if pred_resnet.item() == pred_custom.item():
    print("\n✓ Обе модели дали ОДИНАКОВЫЙ результат!")
else:
    print("\n✗ Модели дали РАЗНЫЕ результаты")