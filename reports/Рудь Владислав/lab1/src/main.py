import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 25
lr = 1.0  # Adadelta использует высокий lr
num_classes = 10

stl10_classes = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


train_transform = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4469, 0.4399, 0.4066],
                         std=[0.2603, 0.2566, 0.2713])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4469, 0.4399, 0.4066],
                         std=[0.2603, 0.2566, 0.2713])
])

train_dataset = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.STL10(
    root="./data",
    split="test",
    download=True,
    transform=test_transform
)


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)


class SimpleSTL10CNN(nn.Module):
    def __init__(self):
        super(SimpleSTL10CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 × 48 × 48

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 256 × 24 × 24

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 256 × 12 × 12

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


model = SimpleSTL10CNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=lr)


def evaluate():
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss_sum / total, 100 * correct / total



train_losses = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    test_loss, test_acc = evaluate()

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Test Acc: {test_acc:.2f}%")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.title("Accuracy")

plt.show()


def denormalize(img):
    mean = np.array([0.4469, 0.4399, 0.4066])
    std = np.array([0.2603, 0.2566, 0.2713])
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)

model.eval()
idx = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[idx]

with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    _, pred = torch.max(output, 1)

true_class = stl10_classes[label]
pred_class = stl10_classes[pred.item()]

print(f"True: {true_class}, Predicted: {pred_class}")

plt.imshow(denormalize(image))
plt.title(f"True: {true_class}\nPred: {pred_class}")
plt.axis("off")
plt.show()
