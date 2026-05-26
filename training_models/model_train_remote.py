import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, PILToTensor, Resize
from tqdm import tqdm
import numpy as np
# import optuna as opt
from ultralytics import YOLO
import albumentations as A


main_model = YOLO('../parsing_datasets_scripts/yolo26m.pt')

custom_transforms = [
    A.MotionBlur(blur_limit=(7, 15), p=0.4),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.25,
                               contrast_limit=0.25, p=0.4),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                         val_shift_limit=20, p=0.3),
]

# Purely for server training
main_model.train(data='/workspace/diploma_project/data/server_data.yaml',
                 epochs=100, device='0', imgsz=640,
                 augmentations=custom_transforms, patience=20)

metrics = main_model.val(data='/workspace/diploma_project/data'
                              '/server_data.yaml', imgsz=640)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")