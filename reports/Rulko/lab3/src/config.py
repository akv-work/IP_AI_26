from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET_URL = "https://universe.roboflow.com/mohamed-traore-2ekkp/cats-n9b87/dataset/3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXAMPLES = [
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]
