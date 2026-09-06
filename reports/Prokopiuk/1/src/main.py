import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Импортируем наш класс модели и функции обучения из соседних файлов
from model import NeuralNetwork
from train import train_loop, test_loop


def main():
    # 1. Выбор устройства (GPU / CPU)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 2. Преобразование данных (Torchvision v2)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # 3. Загрузка датасета Fashion-MNIST
    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transforms
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transforms
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 4. Инициализация модели, функции потерь и оптимизатора
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    # 5. Главный цикл обучения по эпохам
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loop(test_loader, model, loss_fn, device)

    print("Done!")

    # 6. Сохранение обученной модели
    torch.save(model.state_dict(), "fashion_cnn.pth")
    print("Saved PyTorch Model State to fashion_cnn.pth")


if __name__ == "__main__":
    main()