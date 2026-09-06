import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #1*28*28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            #32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),
            #32*14*14
            nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3, padding=1),
            #64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),
            #64*7*7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = NeuralNetwork()
    input = torch.randn(64, 1, 28, 28)
    output = model(input)

    print(f"Input: {input.shape}")
    print(f"Output: {output.shape}")