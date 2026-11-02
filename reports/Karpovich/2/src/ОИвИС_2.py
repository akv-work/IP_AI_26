import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)
resnet18 = resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)


epochs = 3
train_losses = []
test_accuracies = []

print("Начало обучения...")

for epoch in range(epochs):
    resnet18.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if (i + 1) % 100 == 0:
            print(f"Эпоха [{epoch + 1}/{epochs}], Шаг [{i + 1}/{len(train_loader)}], Текущий Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)


    resnet18.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f"--- Итог Эпохи [{epoch + 1}/{epochs}]: Loss = {epoch_loss:.4f} | Test Accuracy = {accuracy:.2f}% ---")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
plt.title('График изменения ошибки (Loss)')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, marker='o', color='green', label='Test Accuracy')
plt.title('График точности (Accuracy)')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()


def test_custom_image(model, image_path, device):
    print(f"\nАнализируем изображение: {image_path}")
    model.eval()

    try:
        img = Image.open(image_path).convert('L')

        if img.getpixel((0, 0)) > 127:
            img = ImageOps.invert(img)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f'Нейросеть думает, что это: {predicted.item()}', fontsize=14)
        plt.axis('off')
        plt.show()

    except FileNotFoundError:
        print(f"Ошибка: Файл '{image_path}' не найден. Убедитесь, что картинка лежит в папке Labs.")


test_custom_image(resnet18, 'digit.png', device)

# КЛАСС КАСТОМНОЙ СЕТИ ИЗ ЛР №1
class SimpleCNN_LR1(nn.Module):
    def __init__(self):
        super(SimpleCNN_LR1, self).__init__()
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

# ФУНКЦИЯ СРАВНЕНИЯ ДВУХ МОДЕЛЕЙ
def test_custom_image_both_models(resnet_model, custom_model_path, image_path, device):
    print(f"\n{'=' * 60}")
    print(f"ТЕСТИРОВАНИЕ ИЗОБРАЖЕНИЯ: {image_path}")
    print(f"{'=' * 60}")

    try:
        img = Image.open(image_path).convert('L')

        if img.getpixel((0, 0)) > 127:
            img = ImageOps.invert(img)
            print("✓ Фон был белым - изображение инвертировано")

        # Трансформации для ResNet18
        transform_resnet = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Трансформации для кастомной сети
        transform_custom = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # ===== ResNet18 =====
        img_resnet = transform_resnet(img).unsqueeze(0).to(device)
        resnet_model.eval()
        with torch.no_grad():
            output_resnet = resnet_model(img_resnet)
            _, pred_resnet = torch.max(output_resnet, 1)
        print(f"\nResNet18: {pred_resnet.item()}")

        custom_model = SimpleCNN_LR1().to(device)

        try:
            custom_model.load_state_dict(torch.load(custom_model_path, map_location=device))
            print(f" Загружены веса из: {custom_model_path}")
        except FileNotFoundError:
            print(f"Файл {custom_model_path} не найден!")
            print("  Сначала выполните ЛР №1 и сохраните модель")
            return

        custom_model.eval()
        img_custom = transform_custom(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_custom = custom_model(img_custom)
            _, pred_custom = torch.max(output_custom, 1)
        print(f"Custom CNN (ЛР1): {pred_custom.item()}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Оригинал', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(img, cmap='gray')
        axes[1].set_title(f'ResNet18\nпредсказывает: {pred_resnet.item()}',
                          fontsize=10, color='blue', fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(img, cmap='gray')
        axes[2].set_title(f'Custom CNN (ЛР1)\nпредсказывает: {pred_custom.item()}',
                          fontsize=10, color='green', fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f'Сравнение моделей', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        if pred_resnet.item() == pred_custom.item():
            print("\nОбе модели дали ОДИНАКОВЫЙ результат!")
        else:
            print("Модели дали РАЗНЫЕ результаты")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{image_path}' не найден!")

test_custom_image_both_models(resnet18, 'custom_cnn_lr1.pth', 'digit.png', device)