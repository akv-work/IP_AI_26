import matplotlib
matplotlib.use("TkAgg")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


train_losses = []
test_losses = []
test_accuracies = []


def train_epoch():
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return epoch_loss, accuracy


print("Начинаем обучение...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch()
    test_loss, test_acc = evaluate()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Эпоха {epoch}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f} | "
          f"Test loss: {test_loss:.4f} | "
          f"Test accuracy: {test_acc*100:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train loss")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test loss")
plt.xlabel("Эпоха")
plt.ylabel("Loss (CrossEntropy)")
plt.title("Изменение ошибки в процессе обучения (MNIST, SGD)")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), [acc * 100 for acc in test_accuracies],
         label="Test accuracy", color="green", marker="o")
plt.xlabel("Эпоха")
plt.ylabel("Точность, %")
plt.title("Изменение точности на тестовой выборке (MNIST, SGD)")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.show()

print(f"\nИтоговая точность на тестовой выборке: {test_accuracies[-1]*100:.2f}%")

def visualize_prediction(index=0):
    model.eval()
    image, true_label = test_dataset[index]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    img_np = image.squeeze().numpy() * 0.3081 + 0.1307

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap="gray")
    plt.title(f"Истинный класс: {true_label}\n"
              f"Предсказание: {predicted_label} (уверенность {confidence*100:.1f}%)")
    plt.axis("off")
    plt.savefig("prediction_example.png")
    plt.show()


random_index = np.random.randint(0, len(test_dataset))
visualize_prediction(random_index)

def visualize_custom_photo(path):

    model.eval()

    # открываем и переводим в градации серого
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img).astype(np.float32) / 255.0

    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    img_norm = (img_array - 0.1307) / 0.3081
    input_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Своя фотография\n"
              f"Предсказание: {predicted_label} (уверенность {confidence*100:.1f}%)")
    plt.axis("off")
    plt.savefig("prediction_custom_photo.png")
    plt.show()

    print(f"Своя фотография ({path}): предсказан класс {predicted_label}, "
          f"уверенность {confidence*100:.1f}%")


custom_photo_candidates = ["my_photo.png", "my_photo.jpg", "my_photo.jpeg"]
custom_photo_path = None
for candidate in custom_photo_candidates:
    if os.path.exists(candidate):
        custom_photo_path = candidate
        break

if custom_photo_path is not None:
    print(f"\nНайдена своя фотография: {custom_photo_path}, проверяем на модели...")
    visualize_custom_photo(custom_photo_path)
else:
    print("\nСвоя фотография (my_photo.png / .jpg) не найдена в папке - пропускаем этот шаг.")

torch.save(model.state_dict(), "mnist_cnn_sgd.pth")
print("Модель сохранена в mnist_cnn_sgd.pth")
