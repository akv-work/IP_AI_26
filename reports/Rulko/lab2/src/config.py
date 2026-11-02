import random

import numpy as np
import torch

SEED = 42

NUM_CLASSES = 100
CLASSES = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
           'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
           'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
           'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
           'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
           'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
           'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
           'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
           'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
           'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
           'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
           'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
           'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
           'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
           'worm')

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_ROOT = './data'
NUM_WORKERS = 2

BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
N_EPOCHS = 8
LR = 1e-3

BATCH_SIZE_PT = 64
TEST_BATCH_SIZE_PT = 128
N_EPOCHS_PT = 3
LR_PT = 1e-3

TEST_IMAGE_PATH = r'C:\Users\rulko\Desktop\test1.png'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
