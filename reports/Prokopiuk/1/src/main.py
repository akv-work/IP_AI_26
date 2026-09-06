import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import NeuralNetwork
from train import train_loop, test_loop

if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transforms
    )
    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transforms
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loop(test_loader, model, loss_fn, device)

    torch.save(model.state_dict(), "fashion_cnn.pth")
    print("Saved PyTorch Model State to fashion_cnn.pth")