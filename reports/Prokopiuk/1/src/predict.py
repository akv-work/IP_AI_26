import random
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import v2

from model import NeuralNetwork

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def predict_random_sample():
  device = (
      torch.accelerator.current_accelerator().type
      if torch.accelerator.is_available()
      else "cpu"
  )

  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load("fashion_cnn.pth", map_location=device))
  model.eval()

  transforms = v2.Compose(
      [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
  )
  test_data = datasets.FashionMNIST(
      root="data", train=False, download=True, transform=transforms
  )

  idx = random.randint(0, len(test_data) - 1)
  image, true_label = test_data[idx]

  input_tensor = image.unsqueeze(0).to(device)
  with torch.no_grad():
    output = model(input_tensor)
    pred_label = output.argmax(1).item()

  print(f"Real class: {CLASSES[true_label]}")
  print(f"Predicted class: {CLASSES[pred_label]}")

  plt.imshow(image.squeeze(), cmap="gray")
  plt.title(f"True: {CLASSES[true_label]}\nPred: {CLASSES[pred_label]}")
  plt.axis("off")
  plt.show()


if __name__ == "__main__":
  predict_random_sample()