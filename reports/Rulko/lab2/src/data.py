import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config


def _make_loaders(train_tf, test_tf, batch_size, test_batch_size):
    train_set = torchvision.datasets.CIFAR100(root=config.DATA_ROOT, train=True,
                                              download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root=config.DATA_ROOT, train=False,
                                             download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    print(f"Train: {len(train_set)}, Test: {len(test_set)}, "
          f"размер изображения: {train_set[0][0].shape}")
    return train_loader, test_loader


def get_custom_loaders():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return _make_loaders(train_tf, test_tf, config.BATCH_SIZE, config.TEST_BATCH_SIZE)


def get_pretrained_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return _make_loaders(train_tf, test_tf, config.BATCH_SIZE_PT, config.TEST_BATCH_SIZE_PT)
