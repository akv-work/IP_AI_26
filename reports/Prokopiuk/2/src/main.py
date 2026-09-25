import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pretrained_model import create_model
from train import train_loop, test_loop

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=3),
    v2.ToDtype(torch.float32, scale=True)
])

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

model = create_model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=1.0)

history = {
    "train_loss": [],
    "test_loss": [],
    "test_acc": []
}

epochs = 5

for t in range(epochs):
    print(f"Epoch {t + 1}\n")
    train_loss = train_loop(
        train_loader,
        model,
        loss_fn,
        optimizer,
        device
    )
    test_loss, test_acc = test_loop(
        test_loader,
        model,
        loss_fn,
        device
    )
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

torch.save(model.state_dict(), "mobilenet_fashion.pth")
print("Saved PyTorch Model State to mobilenet_fashion.pth")

plt.figure()
plt.plot(history["train_loss"], label="Train loss")
plt.plot(history["test_loss"], label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and test loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(history["test_acc"], label="Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test accuracy")
plt.legend()
plt.grid()
plt.show()