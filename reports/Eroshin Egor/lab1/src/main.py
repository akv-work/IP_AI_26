import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = 'cifar10_model.pth'  # Файл для сохранения весов
FORCE_RETRAIN = False             # Поставить True, если нужно будет переобучить


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def imshow(img, title):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

if __name__ == '__main__':
    torch.manual_seed(42)

    # 1. ДАННЫЕ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)


    if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Загружена ранее обученная модель из файла '{MODEL_PATH}'. Обучение пропущено.")
    else:

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        epochs = 20
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        print("Начало обучения...")
        for epoch in range(epochs):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(trainloader.dataset)
            epoch_train_acc = 100 * correct_train / total_train

            model.eval()
            val_loss, correct_test, total_test = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            epoch_test_loss = val_loss / len(testloader.dataset)
            epoch_test_acc = 100 * correct_test / total_test

            train_losses.append(epoch_train_loss)
            test_losses.append(epoch_test_loss)
            train_accs.append(epoch_train_acc)
            test_accs.append(epoch_test_acc)

            print(f"Эпоха {epoch+1:02d}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
                  f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")


        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Обучение завершено. Модель сохранена в '{MODEL_PATH}'.")


        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
        plt.title('Изменение ошибки (Loss)')
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
        plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy')
        plt.title('Изменение точности (Accuracy)')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()


    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"\nИтоговая точность на тестовой выборке: {100 * correct / total:.2f}%")


    dataiter = iter(testloader)
    images, labels = next(dataiter)
    idx = np.random.randint(0, len(images))
    img = images[idx]
    true_label = classes[labels[idx]]

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class = torch.max(output, 1)

    pred_label = classes[predicted_class.item()]

    plt.figure(figsize=(4, 4))
    imshow(img.cpu(), f"Истинный: {true_label}\nПредсказанный: {pred_label}")
    plt.savefig('inference_example.png')
    plt.show()

    print("\nВероятности по классам:")
    for i, prob in enumerate(probabilities):
        print(f"{classes[i]:12s}: {prob.item() * 100:.2f}%")