import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import urllib.request
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Загрузка датасета MNIST...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
test_dataset = torch.utils.data.Subset(test_dataset, range(2000))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Загрузка предобученной модели AlexNet...")
model = models.alexnet(weights=None)
model.load_state_dict(torch.load('alexnet-owt-7be5be79.pth'))

model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(9216, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

epochs = 3
loss_history = []

print("Начало обучения...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"--- Эпоха {epoch + 1}/{epochs} ---")
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 10 == 0:
            print(f"  Батч {i}/{len(train_loader)}, Ошибка: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f"Эпоха {epoch + 1} завершена. Средняя ошибка: {epoch_loss:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.title('График изменения ошибки')
plt.grid(True)
plt.show()

print("Оценка на тестовой выборке...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Точность на тестовой выборке: {100 * correct / total:.2f}%')


def visualize_prediction(image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    img = image_tensor.squeeze().cpu().numpy()
    img = img * 0.3081 + 0.1307
    plt.imshow(img, cmap='gray')
    plt.title(f'Предсказание: {predicted.item()}')
    plt.axis('off')
    plt.show()


print("Визуализация случайного изображения...")
url = "https://raw.githubusercontent.com/rhammell/mnist-dataset/master/test/7/00001.png"
try:
    with urllib.request.urlopen(url) as response:
        img = Image.open(io.BytesIO(response.read()))
    img_tensor = transform(img)
    visualize_prediction(img_tensor)
except Exception as e:
    print(f"Не удалось загрузить изображение: {e}")
    sample_image, sample_label = test_dataset[0]
    print(f'Использую изображение из тестовой выборки. Истинная метка: {sample_label}')
    visualize_prediction(sample_image)