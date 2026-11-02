import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_data_loaders(batch_size=64, data_root="./data"):
    train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset
