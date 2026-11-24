"""ЛР2 — Яшков П. А., вариант 4. CIFAR-100 / SGD / MobileNetV3 Small.

Воспроизводимый независимый запуск эксперимента Colab в локальной папке.
Оригинальный блокнот и исходные результаты 10 эпох прилагаются отдельно.
"""
import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 MobileNetV3 Small + SGD')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data', type=Path, default=Path('./data'))
    parser.add_argument('--out', type=Path, default=Path('./results_repeated'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'plots').mkdir(exist_ok=True)
    model_dir = args.out / 'models'
    model_dir.mkdir(exist_ok=True)
    data_dir = args.data

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_data = datasets.CIFAR100(str(data_dir), train=True,
                                   download=True, transform=train_transform)
    test_data = datasets.CIFAR100(str(data_dir), train=False,
                                  download=True, transform=test_transform)
    # В оригинальном Colab применялись 2 воркера. При ошибках завершения
    # процессов DataLoader допустимо установить num_workers=0.
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              pin_memory=device.type == 'cuda')
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=device.type == 'cuda')

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    dim = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(dim, 100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_path = model_dir / 'mobilenet_v3_best.pth'
    last_path = model_dir / 'mobilenet_v3_last_checkpoint.pth'
    hist = []
    best_accuracy = 0.0
    start_epoch = 0
    if args.resume and last_path.exists():
        cp = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
        hist = cp['history']
        best_accuracy = cp['best_accuracy']
        start_epoch = cp['epoch']
        print('Продолжение с эпохи:', start_epoch + 1)

    print('Устройство:', device, '| эпох:', args.epochs,
          '| параметров:', sum(p.numel() for p in model.parameters()))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Эпоха {epoch + 1}/{args.epochs}'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = labels.size(0)
            train_loss += loss.item() * n
            train_correct += (output.argmax(1) == labels).sum().item()
            train_total += n
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    output = model(images)
                    loss = criterion(output, labels)
                n = labels.size(0)
                test_loss += loss.item() * n
                test_correct += (output.argmax(1) == labels).sum().item()
                test_total += n
        row = dict(epoch=epoch + 1,
                   train_loss=train_loss / train_total,
                   train_accuracy=100 * train_correct / train_total,
                   test_loss=test_loss / test_total,
                   test_accuracy=100 * test_correct / test_total)
        hist.append(row)
        print(f"Эпоха {epoch+1}: train loss={row['train_loss']:.4f}, "
              f"train acc={row['train_accuracy']:.2f}%, "
              f"test loss={row['test_loss']:.4f}, "
              f"test acc={row['test_accuracy']:.2f}%")
        if row['test_accuracy'] > best_accuracy:
            best_accuracy = row['test_accuracy']
            torch.save(model.state_dict(), best_path)
        scheduler.step()
        torch.save({'epoch': epoch+1, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': hist, 'best_accuracy': best_accuracy}, last_path)
        with (args.out / 'mobilenet_history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(hist)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [r['epoch'] for r in hist]
    for key, axis, label in [('loss', axes[0], 'CrossEntropyLoss'),
                             ('accuracy', axes[1], 'Точность, %')]:
        axis.plot(epochs, [r['train_' + key] for r in hist], marker='o', label='Обучение')
        axis.plot(epochs, [r['test_' + key] for r in hist], marker='o', label='Тест')
        axis.set_title('MobileNetV3: ' + ('Ошибка' if key=='loss' else 'Точность'))
        axis.set_xlabel('Эпоха')
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.out / 'plots' / 'mobilenet_training.png', dpi=200)
    plt.close(fig)
    print('Лучшая тестовая точность:', round(best_accuracy, 2), '%')
    print('Результаты сохранены:', args.out)


if __name__ == '__main__':
    main()
