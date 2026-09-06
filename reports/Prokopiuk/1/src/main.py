import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from model import NeuralNetwork
from train import train_loop, test_loop

def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    plt.plot(epochs, history["test_loss"], "o-", label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs, history["test_acc"], "s-", color="green", label="Test Accuracy"
    )
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=300)
    print("Saved in 'learning_curves.png'")
    plt.show()

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

    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_loop(test_loader, model, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    torch.save(model.state_dict(), "fashion_cnn.pth")
    print("Saved PyTorch Model State to fashion_cnn.pth")

    plot_metrics(history)