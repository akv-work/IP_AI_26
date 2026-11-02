!mkdir -p ./data
!wget -q -O ./data/cifar10.tgz https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
!tar -xzf ./data/cifar10.tgz -C ./data/

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121, DenseNet121_Weights
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_cpu = transforms.Compose([
    transforms.ToTensor()
])

trainset = ImageFolder(root='./data/cifar10/train', transform=transform_cpu)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

testset = ImageFolder(root='./data/cifar10/test', transform=transform_cpu)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

classes = trainset.classes

gpu_transforms = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

weights = DenseNet121_Weights.DEFAULT
net = densenet121(weights=weights)

num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(num_ftrs, 10)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(net.parameters())
epochs = 5
loss_history = []

print("\nНачало обучения...")
for epoch in range(epochs):
    running_loss = 0.0
    net.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = gpu_transforms(inputs)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    loss_history.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='b')
plt.title('DenseNet121 Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss (Cross Entropy)')
plt.grid(True)
plt.show()

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = gpu_transforms(inputs)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'\nAccuracy on test images: {accuracy:.2f}%')

def predict_custom_image(image_path, model):
    model.eval()
    image = Image.open(image_path).convert('RGB')

    single_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = single_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_name = classes[predicted.item()]

    plt.imshow(image)
    plt.title(f'Prediction: {class_name}')
    plt.axis('off')
    plt.show()

    return class_name


IMAGE_PATH = 'frog_sample_1.png'
TRUE_LABEL = 'frog'

def predict_and_visualize(image_path, true_label, model, class_names):
    model.eval()

    image = Image.open(image_path).convert('RGB')

    single_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = single_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)

    predicted_label = class_names[predicted_idx.item()]

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"True: {true_label}\nPredicted: {predicted_label}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

predict_and_visualize(IMAGE_PATH, TRUE_LABEL, net, classes)
