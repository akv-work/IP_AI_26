import torch

from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def create_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)

    return model

if __name__ == "__main__":
    model = create_model()
    input = torch.randn(64, 3, 224, 224)
    output = model(input)

    print(f"Input: {input.shape}")
    print(f"Output: {output.shape}")