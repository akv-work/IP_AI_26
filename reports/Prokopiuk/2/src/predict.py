import random
import torch
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from pretrained_model import create_model
from model import NeuralNetwork

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

custom_model = NeuralNetwork().to(device)
custom_model.load_state_dict(
    torch.load("fashion_cnn.pth", map_location=device)
)
custom_model.eval()

pretrained_model = create_model().to(device)
pretrained_model.load_state_dict(
    torch.load("mobilenet_fashion.pth", map_location=device)
)
pretrained_model.eval()

index = random.randrange(len(test_data))
image, label = test_data[index]

custom_image = image.unsqueeze(0).to(device)

pretrained_image = image.repeat(3, 1, 1).unsqueeze(0).to(device)
pretrained_image = torch.nn.functional.interpolate(
    pretrained_image,
    size=(224, 224),
    mode="bilinear"
)

with torch.no_grad():
    custom_output = custom_model(custom_image)
    pretrained_output = pretrained_model(pretrained_image)

custom_prediction = custom_output.argmax(1).item()
pretrained_prediction = pretrained_output.argmax(1).item()

print(f"Real class: {classes[label]}")
print(f"Custom CNN: {classes[custom_prediction]}")
print(f"MobileNet v3: {classes[pretrained_prediction]}")

plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.title(
    f"Real: {classes[label]}\n"
    f"Custom CNN: {classes[custom_prediction]}\n"
    f"MobileNet v3: {classes[pretrained_prediction]}"
)
plt.show()