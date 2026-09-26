import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_loss(epoch_losses):
    num_epochs = len(epoch_losses)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="y", color="r", label="Train Loss")
    plt.title("Loss Function Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


def show_prediction(model, test_dataset):
    model.eval()
    sample_idx = np.random.randint(0, len(test_dataset))
    image, true_label = test_dataset[sample_idx]
    input_tensor = image.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output, 1)

    plt.figure(figsize=(4, 4))
    img_display = image.squeeze().numpy() * 0.5 + 0.5
    plt.imshow(img_display, cmap="gray")
    plt.title(f"Real: {true_label} | Predict: {predicted_label.item()}")
    plt.axis("off")
    plt.show()
