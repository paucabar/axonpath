import torch
import torch.nn as nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import copy

def transforms_fn(img_height, img_width):
    train_transform = A.Compose(
        [
            A.RandomScale(scale_limit=0.25, interpolation=cv2.INTER_LINEAR, p=0.25),
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
            A.RandomCrop(height=img_height, width=img_width),
            ToTensorV2(),
        ],
    )
    
    return train_transform, val_transforms

def visualize_augmentations(dataset, idx=0, samples=20, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 16))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def post_trans_fn(pred):
    m = nn.Softmax(dim=None)
    smax = m(pred)
    argmax = torch.argmax(smax, dim=1)
    return smax, argmax