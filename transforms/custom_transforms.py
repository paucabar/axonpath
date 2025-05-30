import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def transforms_fn(img_height, img_width):
    train_transform = A.Compose(
        [
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.25),
            A.RandomCrop(height=img_height, width=img_width, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GaussianBlur (blur_limit=(1, 5), sigma_limit=0, always_apply=False, p=0.25),
                A.MedianBlur (blur_limit=3, always_apply=False, p=0.25),
                A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.25),
                A.Defocus (radius=(1, 5), alias_blur=(0.1, 0.5), always_apply=False, p=0.25)
            ], p=0.25),
            A.ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0., always_apply=False, p=0.25),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.CenterCrop(
                height = img_height,
                width = img_width,
                p = 1.0
            ),
            ToTensorV2(),
        ],
    )
    
    return train_transform, val_transforms