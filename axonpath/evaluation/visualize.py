"""Overlay utilities for visualising AxonPath predictions on raw images.

Semantic overlay: yellow for myelin (class 1), cyan for inner cylinder (class 2).
Instance overlay: Glasbey-style hue-spaced colours per fibre instance.

Both use the same alpha and blending formula so semantic and instance outputs
can be compared directly side-by-side.
"""

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA = 140 / 255

SEM_COLOURS = {
    1: (255, 220,   0),  # myelin: yellow
    2: (  0, 220, 255),  # inner cylinder: cyan
}


def to_display(image: np.ndarray) -> np.ndarray:
    """Convert a raw tile image (float 0–1 or uint8 0–255) to uint8."""
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255
    return img.clip(0, 255).astype(np.uint8)


def glasbey_colors(n: int) -> list[tuple[int, int, int]]:
    """N evenly-spaced hues at high saturation/value — bright, distinct colours."""
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95))
        for i in range(n)
    ]


def semantic_overlay(image: np.ndarray, pred_sem: np.ndarray) -> np.ndarray:
    """Return an RGB array with myelin/inner-cylinder regions colour-blended onto the image."""
    canvas = np.stack([to_display(image)] * 3, axis=-1).astype(np.float32)
    for cls, (r, g, b) in SEM_COLOURS.items():
        region = pred_sem == cls
        canvas[region] = canvas[region] * (1 - ALPHA) + np.array([r, g, b]) * ALPHA
    return canvas.clip(0, 255).astype(np.uint8)


def instance_overlay(image: np.ndarray, pred_fibre: np.ndarray) -> np.ndarray:
    """Return an RGB array with each fibre instance drawn in a distinct Glasbey colour."""
    canvas = np.stack([to_display(image)] * 3, axis=-1).astype(np.float32)
    unique_ids = np.unique(pred_fibre)
    unique_ids = unique_ids[unique_ids != 0]
    colors = glasbey_colors(len(unique_ids))
    for i, label_id in enumerate(unique_ids):
        color = np.array(colors[i], dtype=np.float32)
        canvas[pred_fibre == label_id] = (
            canvas[pred_fibre == label_id] * (1 - ALPHA) + color * ALPHA
        )
    return canvas.clip(0, 255).astype(np.uint8)


def save_semantic_overlay(image: np.ndarray, pred_sem: np.ndarray, out_path: Path) -> None:
    Image.fromarray(semantic_overlay(image, pred_sem)).save(out_path)


def save_instance_overlay(image: np.ndarray, pred_fibre: np.ndarray, out_path: Path) -> None:
    Image.fromarray(instance_overlay(image, pred_fibre)).save(out_path)
