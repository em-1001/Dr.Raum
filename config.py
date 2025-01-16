# config.py

import numpy as np
import torch
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)

rand_affine = RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=0.7,
    spatial_size=(96, 128, 128),
    translate_range=(10, 20, 20),
    rotate_range=(np.pi / 72, np.pi / 72, np.pi / 72),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)

TRAIN_DATASET_PATH = "/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
SAVE_PATH = "/content/drive/MyDrive/unetr/unter_weights.pth"

# Define seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3
}

# Select Slices and Image Size
VOLUME_SLICES = 96
VOLUME_START_AT = 25 # first slice of volume that we will include
IMG_SIZE=128
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL  = False

