"""Классификация отдельного изображения сохранённой MobileNetV3.

Пример: python predict_image.py path/to/photo.jpg --weights ../models/mobilenet_v3_best.pth
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = ('apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy '
'bridge bus butterfly camel can castle caterpillar cattle chair chimpanzee clock cloud '
'cockroach couch crab crocodile cup dinosaur dolphin elephant flatfish forest fox girl '
'hamster house kangaroo keyboard lamp lawn_mower leopard lion lizard lobster man maple_tree '
'motorcycle mountain mouse mushroom oak_tree orange orchid otter palm_tree pear pickup_truck '
'pine_tree plain plate poppy porcupine possum rabbit raccoon ray road rocket rose sea seal '
'shark shrew skunk skyscraper snail snake spider squirrel streetcar sunflower sweet_pepper '
'table tank telephone television tiger tractor train trout tulip turtle wardrobe whale willow_tree '
'wolf woman worm').split()
assert len(CLASSES) == 100


def predict(image_file, weights_file, save_file=None):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 100)
    state = torch.load(weights_file, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0))
        prob = logits.softmax(1).squeeze(0)
        values, indices = prob.topk(5)
    names = [CLASSES[i] for i in indices.tolist()]
    print('Распознано:', names[0], f'{values[0].item()*100:.2f}%')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Исходное изображение')
    axes[1].barh(names[::-1], [x*100 for x in values.tolist()[::-1]])
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Вероятность, %')
    axes[1].set_title('Топ-5 классов CIFAR-100')
    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=180)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--weights', type=Path,
                        default=Path(__file__).resolve().parent.parent/'models/mobilenet_v3_best.pth')
    parser.add_argument('--save', type=Path, default=None)
    args = parser.parse_args()
    predict(args.image, args.weights, args.save)
