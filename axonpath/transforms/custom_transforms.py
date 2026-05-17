"""
Custom augmentation transforms replacing albumentations.

All spatial transforms (Rotate, RandomScale, CenterCrop, RandomCrop, Horizontal/VerticalFlip)
are applied jointly to the image and all masks. Intensity transforms (GaussianBlur, MedianBlur,
GaussNoise, Defocus, ColorJitter, RandomGamma) are applied to the image only.

Interface contract (identical to albumentations):
    transformed = transform(image=image, masks=masks)
    image = transformed["image"]
    masks = transformed["masks"]

Where `image` is a float32 HxW or HxWxC numpy array and `masks` is a list of 2D numpy
arrays (HxW). After ToTensor, image becomes a float32 CxHxW torch.Tensor.
"""

import random
import numpy as np
import torch
from scipy import ndimage


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks):
        # Accept both positional and keyword-argument call styles:
        #   transform(image, masks)          — internal chaining
        #   transform(image=img, masks=msks) — dataset call site
        for t in self.transforms:
            image, masks = t(image, masks)
        return {"image": image, "masks": masks}


class OneOf:
    """With probability p, apply one randomly selected transform from the list."""

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        return random.choice(self.transforms)(image, masks)


# ---------------------------------------------------------------------------
# Spatial transforms (image + masks jointly)
# ---------------------------------------------------------------------------

class Rotate:
    """
    Rotate image and masks by a random angle sampled from [-limit, limit] degrees.

    fill_values: one constant fill value per mask, in dataset mask order
        [mask_fibre, mask_axon, mask_sem, sdt_fibre, sdt_axon].
    Integer label masks are rotated with order=0 (nearest-neighbour) to avoid
    interpolation artefacts; float SDT maps use order=1 (bilinear).
    """

    def __init__(self, limit, p=0.5, fill_values=None):
        self.limit = limit
        self.p = p
        self.fill_values = fill_values or []

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        angle = random.uniform(-self.limit, self.limit)

        if image.ndim == 3:
            rotated_image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1, cval=0.0)
        else:
            rotated_image = ndimage.rotate(image, angle, reshape=False, order=1, cval=0.0)

        rotated_masks = []
        for i, mask in enumerate(masks):
            fill = float(self.fill_values[i]) if i < len(self.fill_values) else 0.0
            order = 0 if np.issubdtype(mask.dtype, np.integer) else 1
            rotated_masks.append(
                ndimage.rotate(mask, angle, reshape=False, order=order, cval=fill)
            )
        return rotated_image, rotated_masks


class CenterCrop:
    """Crop the central (height, width) region."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, masks):
        h, w = image.shape[:2]
        assert h >= self.height and w >= self.width, (
            f"CenterCrop({self.height}, {self.width}) exceeds image size ({h}, {w})"
        )
        top = (h - self.height) // 2
        left = (w - self.width) // 2
        image = image[top:top + self.height, left:left + self.width]
        masks = [m[top:top + self.height, left:left + self.width] for m in masks]
        return image, masks


class RandomCrop:
    """Crop a randomly positioned (height, width) region."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, masks):
        h, w = image.shape[:2]
        assert h >= self.height and w >= self.width, (
            f"RandomCrop({self.height}, {self.width}) exceeds image size ({h}, {w})"
        )
        top = random.randint(0, h - self.height)
        left = random.randint(0, w - self.width)
        image = image[top:top + self.height, left:left + self.width]
        masks = [m[top:top + self.height, left:left + self.width] for m in masks]
        return image, masks


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        image = np.flip(image, axis=1).copy()
        masks = [np.flip(m, axis=1).copy() for m in masks]
        return image, masks


class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        image = np.flip(image, axis=0).copy()
        masks = [np.flip(m, axis=0).copy() for m in masks]
        return image, masks


class ElasticDeformation:
    """
    Apply random elastic deformation to image and masks using a shared displacement field.

    A smooth displacement field is generated by filtering uniform random noise with a
    Gaussian (sigma controls smoothness) and scaling by alpha (controls magnitude in px).
    The same field is applied to the image and all masks; integer masks use nearest-
    neighbour interpolation, float SDT maps use bilinear — identical to Rotate.

    Typical values for 512×512 EM tiles: alpha=100 (≈12 px max displacement), sigma=6.
    """

    def __init__(self, alpha=34, sigma=6, p=0.25, fill_values=None):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.fill_values = fill_values or []

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks

        h, w = image.shape[:2]
        # Smooth random displacement fields in x and y
        dx = ndimage.gaussian_filter(
            np.random.uniform(-1, 1, (h, w)), sigma=self.sigma
        ) * self.alpha
        dy = ndimage.gaussian_filter(
            np.random.uniform(-1, 1, (h, w)), sigma=self.sigma
        ) * self.alpha

        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords_y = np.clip(yy + dy, 0, h - 1)
        coords_x = np.clip(xx + dx, 0, w - 1)
        coords = [coords_y.ravel(), coords_x.ravel()]

        if image.ndim == 2:
            deformed_image = ndimage.map_coordinates(image, coords, order=1, mode="nearest")
            deformed_image = deformed_image.reshape(h, w).astype(np.float32)
        else:
            channels = [
                ndimage.map_coordinates(image[..., c], coords, order=1, mode="nearest").reshape(h, w)
                for c in range(image.shape[2])
            ]
            deformed_image = np.stack(channels, axis=-1).astype(np.float32)

        deformed_masks = []
        for i, mask in enumerate(masks):
            order = 0 if np.issubdtype(mask.dtype, np.integer) else 1
            dm = ndimage.map_coordinates(mask, coords, order=order, mode="nearest")
            deformed_masks.append(dm.reshape(h, w).astype(mask.dtype))

        return deformed_image, deformed_masks


class RandomScale:
    """
    Randomly scale the image and masks by a factor sampled from scale_range.

    After downscaling the image is padded back to its original tile dimensions
    (image: 0, integer masks: 0, SDT maps: -1) so the subsequent RandomCrop
    always has enough data to work with. After upscaling, RandomCrop handles
    the larger array naturally.

    fill_values: one fill value per mask, same order as Rotate.
    """

    def __init__(self, scale_range=(0.9, 1.1), p=0.5, fill_values=None):
        self.scale_range = scale_range
        self.p = p
        self.fill_values = fill_values or []

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks

        scale = random.uniform(*self.scale_range)
        orig_h, orig_w = image.shape[:2]

        zoom = (scale, scale) if image.ndim == 2 else (scale, scale, 1.0)
        scaled_image = ndimage.zoom(image, zoom, order=1, cval=0.0)

        scaled_masks = []
        for i, mask in enumerate(masks):
            fill = float(self.fill_values[i]) if i < len(self.fill_values) else 0.0
            order = 0 if np.issubdtype(mask.dtype, np.integer) else 1
            scaled_masks.append(ndimage.zoom(mask, (scale, scale), order=order, cval=fill))

        # Pad to original size when downscaled
        new_h, new_w = scaled_image.shape[:2]
        if new_h < orig_h or new_w < orig_w:
            pad_h = orig_h - new_h
            pad_w = orig_w - new_w
            pad_img = ((0, pad_h), (0, pad_w)) if image.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
            scaled_image = np.pad(scaled_image, pad_img, constant_values=0.0)
            padded_masks = []
            for i, mask in enumerate(scaled_masks):
                fill = self.fill_values[i] if i < len(self.fill_values) else 0
                padded_masks.append(np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=fill))
            scaled_masks = padded_masks

        return scaled_image, scaled_masks


# ---------------------------------------------------------------------------
# Intensity transforms (image only)
# ---------------------------------------------------------------------------

class GaussianBlur:
    """Gaussian blur with sigma sampled uniformly from sigma_range."""

    def __init__(self, sigma_range=(0.5, 2.0), p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        sigma = random.uniform(*self.sigma_range)
        if image.ndim == 2:
            blurred = ndimage.gaussian_filter(image, sigma=sigma)
        else:
            blurred = np.stack(
                [ndimage.gaussian_filter(image[..., c], sigma=sigma) for c in range(image.shape[2])],
                axis=-1,
            )
        return blurred.astype(np.float32), masks


class MedianBlur:
    """Median filter with the given kernel size."""

    def __init__(self, size=3, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        if image.ndim == 2:
            filtered = ndimage.median_filter(image, size=self.size)
        else:
            filtered = np.stack(
                [ndimage.median_filter(image[..., c], size=self.size) for c in range(image.shape[2])],
                axis=-1,
            )
        return filtered.astype(np.float32), masks


class GaussNoise:
    """Add Gaussian noise with variance sampled uniformly from var_limit."""

    def __init__(self, var_limit=(0.002, 0.01), p=0.5):
        self.var_limit = var_limit
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        var = random.uniform(*self.var_limit)
        noise = np.random.normal(0.0, var ** 0.5, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0).astype(np.float32), masks


class Defocus:
    """
    Approximate out-of-focus blur using a uniform (box) filter.

    A box blur is a better approximation of optical defocus than a Gaussian,
    and is genuinely distinct from GaussianBlur inside OneOf.
    size_range: odd integers to sample from (e.g. 3, 5, 7).
    """

    def __init__(self, size_range=(3, 7), p=0.5):
        self.size_range = size_range
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        # Sample an odd kernel size
        lo, hi = self.size_range
        candidates = [s for s in range(lo, hi + 1, 2)]
        size = random.choice(candidates) if candidates else lo
        if image.ndim == 2:
            blurred = ndimage.uniform_filter(image, size=size)
        else:
            blurred = np.stack(
                [ndimage.uniform_filter(image[..., c], size=size) for c in range(image.shape[2])],
                axis=-1,
            )
        return blurred.astype(np.float32), masks


class ColorJitter:
    """
    Randomly adjust brightness and contrast.
    Saturation is omitted — EM images are single-channel grayscale.
    """

    def __init__(self, brightness=0.3, contrast=0.3, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            image = image + delta
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean
        return np.clip(image, 0.0, 1.0).astype(np.float32), masks


class RandomGamma:
    """
    Apply random gamma correction (image ** gamma).
    Simulates variability in detector response — a grayscale analogue of saturation.
    """

    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image, masks):
        if random.random() > self.p:
            return image, masks
        gamma = random.uniform(*self.gamma_range)
        return np.power(np.clip(image, 0.0, 1.0), gamma).astype(np.float32), masks


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

class ToTensor:
    """
    Convert image numpy array to a float32 torch.Tensor.
        HxW   → 1xHxW
        HxWxC → CxHxW
    np.ascontiguousarray handles negative strides from upstream flip transforms.
    Masks remain as numpy arrays and are converted later in the dataset.
    """

    def __call__(self, image, masks):
        if image.ndim == 2:
            image_t = torch.from_numpy(np.ascontiguousarray(image[np.newaxis, ...]))
        else:
            image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return image_t, masks


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

# Fill values per mask in dataset order: [mask_fibre, mask_axon, mask_sem, sdt_fibre, sdt_axon]
# Integer label masks fill with 0; SDT maps fill with -1 (exterior/background).
_MASK_FILL = [0, 0, 0, -1, -1]


def transforms_fn(img_height, img_width):
    """
    Return (train_transform, val_transform) for the given crop size.

    Training augmentations include spatial transforms (scale, rotation, flips,
    random crop), blur/noise variants, colour jitter, and gamma correction.
    Validation uses a deterministic centre crop only.
    """
    train_transform = Compose([
        RandomScale(scale_range=(0.8, 1.4), p=0.5, fill_values=_MASK_FILL),
        Rotate(limit=180, p=0.5, fill_values=_MASK_FILL),
        ElasticDeformation(alpha=100, sigma=6, p=0.25, fill_values=_MASK_FILL),
        RandomCrop(height=img_height, width=img_width),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            GaussianBlur(sigma_range=(0.5, 2.0), p=1.0),
            MedianBlur(size=3, p=1.0),
            GaussNoise(var_limit=(0.002, 0.01), p=1.0),
            Defocus(size_range=(3, 7), p=1.0),
        ], p=0.25),
        ColorJitter(brightness=0.3, contrast=0.3, p=0.25),
        RandomGamma(gamma_range=(0.8, 1.2), p=0.25),
        ToTensor(),
    ])

    val_transform = Compose([
        CenterCrop(height=img_height, width=img_width),
        ToTensor(),
    ])

    return train_transform, val_transform
