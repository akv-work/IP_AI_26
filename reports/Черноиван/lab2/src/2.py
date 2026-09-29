import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS_PRETRAINED = 5
EPOCHS_CUSTOM = 10
LR_PRETRAINED = 1e-4
LR_CUSTOM = 1e-3
NUM_WORKERS = 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)



def get_transforms():
    pre_train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    pre_test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    cus_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    cus_test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return pre_train_tf, pre_test_tf, cus_train_tf, cus_test_tf


def get_loaders(train_tf, test_tf):
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    return train_ds, test_ds, train_loader, test_loader


def build_pretrained_mobilenet(num_classes=10):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = models.mobilenet_v3_large(weights=weights)
    # classifier: Linear(960,1280) -> Hardswish -> Dropout -> Linear(1280,1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)   # 32 -> 16 -> 8
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc="  test ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


def fit(model, name, train_loader, test_loader, epochs, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    print(f"\n=== Обучение: {name} | Adam, lr={lr}, эпох={epochs} ===")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(te_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        print(f"Эпоха {epoch}/{epochs} | train loss {tr_loss:.4f} | test loss {te_loss:.4f} | "
              f"train acc {tr_acc*100:.2f}% | test acc {te_acc*100:.2f}% | {time.time()-t0:.0f} c")
    return model, hist

def plot_history(hist, title, prefix):
    epochs = range(1, len(hist["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hist["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, hist["test_loss"], marker="o", label="Test loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title(f"Изменение ошибки: {title}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in hist["train_acc"]], marker="o", label="Train accuracy")
    plt.plot(epochs, [a * 100 for a in hist["test_acc"]], marker="o", label="Test accuracy")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность, %")
    plt.title(f"Изменение точности: {title}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(hist_pre, hist_cus):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(hist_pre["test_loss"]) + 1), hist_pre["test_loss"],
             marker="o", label="MobileNetV3 (предобученная)")
    plt.plot(range(1, len(hist_cus["test_loss"]) + 1), hist_cus["test_loss"],
             marker="s", label="SimpleCNN (кастомная)")
    plt.xlabel("Эпоха")
    plt.ylabel("Test loss")
    plt.title("Сравнение: ошибка на тесте (CIFAR-10, Adam)")
    plt.legend()
    plt.grid(True)
    plt.savefig("compare_test_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(hist_pre["test_acc"]) + 1), [a * 100 for a in hist_pre["test_acc"]],
             marker="o", label="MobileNetV3 (предобученная)")
    plt.plot(range(1, len(hist_cus["test_acc"]) + 1), [a * 100 for a in hist_cus["test_acc"]],
             marker="s", label="SimpleCNN (кастомная)")
    plt.xlabel("Эпоха")
    plt.ylabel("Test accuracy, %")
    plt.title("Сравнение: точность на тесте (CIFAR-10, Adam)")
    plt.legend()
    plt.grid(True)
    plt.savefig("compare_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def predict_pil(model, pil_img, tf, classes):
    model.eval()
    x = tf(pil_img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    top3 = probs.topk(3)
    top3_str = ", ".join(f"{classes[i]} {p*100:.1f}%" for p, i in zip(top3.values.tolist(),
                                                                       top3.indices.tolist()))
    return classes[idx.item()], conf.item(), top3_str


def visualize_custom_image(path, model_pre, model_cus, pre_test_tf, cus_test_tf, classes):

    img = Image.open(path).convert("RGB")


    cus_tf = transforms.Compose([transforms.Resize((32, 32)), cus_test_tf])
    pre_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), pre_test_tf])

    lbl_pre, conf_pre, top3_pre = predict_pil(model_pre, img, pre_tf, classes)
    lbl_cus, conf_cus, top3_cus = predict_pil(model_cus, img, cus_tf, classes)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"MobileNetV3: {lbl_pre} ({conf_pre*100:.1f}%)\n"
              f"SimpleCNN: {lbl_cus} ({conf_cus*100:.1f}%)")
    plt.savefig("prediction_custom_image.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nИзображение: {path}")
    print(f"  MobileNetV3 (предобученная): {lbl_pre} ({conf_pre*100:.1f}%) | топ-3: {top3_pre}")
    print(f"  SimpleCNN (кастомная):       {lbl_cus} ({conf_cus*100:.1f}%) | топ-3: {top3_cus}")


def visualize_random_test_samples(model_pre, model_cus, test_ds_raw, pre_test_tf, cus_test_tf,
                                  classes, n=6):

    idxs = np.random.choice(len(test_ds_raw), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.6))
    for ax, i in zip(axes, idxs):
        img, true = test_ds_raw[i]
        lp, cp, _ = predict_pil(model_pre, img, pre_test_tf, classes)
        lc, cc, _ = predict_pil(model_cus, img, cus_test_tf, classes)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Истина: {classes[true]}\nMNv3: {lp} ({cp*100:.0f}%)\nCNN: {lc} ({cc*100:.0f}%)",
                     fontsize=8)
    plt.tight_layout()
    plt.savefig("prediction_test_samples.png", dpi=150, bbox_inches="tight")
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="путь к своей картинке для визуализации")
    parser.add_argument("--skip-train", action="store_true",
                        help="не обучать, загрузить сохранённые веса .pth")
    args = parser.parse_args()

    print(f"Используемое устройство: {device}")
    pre_train_tf, pre_test_tf, cus_train_tf, cus_test_tf = get_transforms()

    _, _, pre_train_loader, pre_test_loader = get_loaders(pre_train_tf, pre_test_tf)
    _, _, cus_train_loader, cus_test_loader = get_loaders(cus_train_tf, cus_test_tf)
    raw_test = datasets.CIFAR10("./data", train=False, download=True)
    classes = raw_test.classes

    model_pre = build_pretrained_mobilenet(num_classes=10)
    model_cus = SimpleCNN(num_classes=10)

    if args.skip_train and os.path.exists("mobilenetv3_cifar10_adam.pth") \
            and os.path.exists("simplecnn_cifar10_adam.pth"):
        model_pre.load_state_dict(torch.load("mobilenetv3_cifar10_adam.pth", map_location=device))
        model_cus.load_state_dict(torch.load("simplecnn_cifar10_adam.pth", map_location=device))
        model_pre.to(device)
        model_cus.to(device)
        print("Веса загружены, обучение пропущено.")
    else:

        model_pre, hist_pre = fit(model_pre, "MobileNetV3-Large (pretrained)",
                                  pre_train_loader, pre_test_loader, EPOCHS_PRETRAINED, LR_PRETRAINED)
        plot_history(hist_pre, "MobileNetV3 (CIFAR-10, Adam)", "mobilenetv3")
        torch.save(model_pre.state_dict(), "mobilenetv3_cifar10_adam.pth")

        model_cus, hist_cus = fit(model_cus, "SimpleCNN (custom)",
                                  cus_train_loader, cus_test_loader, EPOCHS_CUSTOM, LR_CUSTOM)
        plot_history(hist_cus, "SimpleCNN (CIFAR-10, Adam)", "simplecnn")
        torch.save(model_cus.state_dict(), "simplecnn_cifar10_adam.pth")

        plot_comparison(hist_pre, hist_cus)

        print("\n=== Итоги ===")
        print(f"MobileNetV3 (предобученная): test acc = {hist_pre['test_acc'][-1]*100:.2f}%")
        print(f"SimpleCNN (кастомная):       test acc = {hist_cus['test_acc'][-1]*100:.2f}%")


    visualize_random_test_samples(model_pre, model_cus, raw_test, pre_test_tf_pil(), cus_test_tf_pil(),
                                  classes)
    if args.image and os.path.exists(args.image):
        visualize_custom_image(args.image, model_pre, model_cus, pre_test_tf, cus_test_tf, classes)
    else:
        print("\nСвоя картинка не указана (--image путь) - шаг пропущен.")


def pre_test_tf_pil():

    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def cus_test_tf_pil():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


if __name__ == "__main__":
    main()