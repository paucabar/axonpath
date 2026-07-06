import torch
import os
import importlib.resources
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
from skimage.io import imread
from skimage.morphology import remove_small_objects
from monai.inferers import sliding_window_inference
from axonpath.utils.model_building import model_fn
from axonpath.utils.image_processing import normalize
from axonpath.inference.post_processing import (
    apply_semantic_segmentation_head,
    segment_instances_from_sdt,
)
from axonpath.utils.label_ops import map_axon_labels_to_fibres


def load_image(img_path: str) -> np.ndarray:
    """Load and normalize a grayscale image (1–99th percentile)."""
    image = normalize(imread(img_path).astype(np.float32))
    return image


def resolve_model_path(model_identifier: str) -> str:
    """
    Resolve a model path for inference.

    - First, try loading a built-in pretrained weight from `axonpath.weights`.
    - If not found, fall back to the provided path.

    Args:
        model_identifier (str): Name or path to the model weights (.pth file).

    Returns:
        str: Resolved absolute path to the weights.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # Direct path takes priority — avoids false matches against package weights
    if os.path.isfile(model_identifier):
        return model_identifier

    # Try built-in package weights (name only, no extension)
    try:
        ref = importlib.resources.files("axonpath.weights").joinpath(model_identifier + ".pth")
        path = str(ref)
        if os.path.isfile(path):
            return path
    except (TypeError, FileNotFoundError):
        pass

    # Dev-tree fallback (running from source without install)
    dev_path = Path(__file__).resolve().parent.parent / "weights" / (model_identifier + ".pth")
    if dev_path.is_file():
        return str(dev_path)

    raise FileNotFoundError(f"Model weights not found for: {model_identifier}")


def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.nn.Module:
    """
    Load a PyTorch model from a .pth weights file.

    Reads the stored norm_type from the checkpoint so the model architecture
    matches the weights exactly.

    Args:
        model_path (str): Path or built-in name of the weights file (.pth).
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded model in eval mode.

    Note:
        This loads a .pth weights file for use with run_inference (MONAI sliding
        window). For the TorchScript .pt export used by the QuPath extension, see
        Pipeline and export_torchscript_model.
    """
    resolved_path = resolve_model_path(model_path)
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=True)
    norm_type = checkpoint["norm_type"]
    model = model_fn(device=device, norm_type=norm_type)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    image: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    roi_size=(512, 512),
    fibre_threshold: float = 0.5,
    axon_threshold: float = 0.5,
    min_diameter: float = 30.0,
    sw_batch_size: int = 1,
    overlap: float = 0.75,
    predict_inner_cylinder: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Run inference on a single image using MONAI sliding window.

    The input image must already be normalised (e.g. via load_image or
    axonpath.utils.image_processing.normalize). Passing a raw unscaled image
    will produce incorrect predictions.

    Model output channels:
        0–2 : semantic logits  (background, fibre, inner_cylinder)
        3   : fibre skeleton-aware distance transform (SDT)
        4   : axon skeleton-aware distance transform (SDT)

    Args:
        image (np.ndarray): Normalised 2D input image.
        model (torch.nn.Module): Loaded PyTorch model (from load_model).
        device (str): Inference device.
        roi_size (tuple): Sliding window tile size.
        fibre_threshold (float): SDT threshold for fibre seed detection.
        axon_threshold (float): SDT threshold for axon seed detection.
        min_diameter (float): Expected minimum fibre diameter in pixels.
            Axon minimum diameter is derived as min_diameter / 2 (axons are
            roughly half the fibre diameter).
        sw_batch_size (int): Number of tiles processed in parallel.
        overlap (float): Fractional overlap between adjacent sliding windows.
        predict_inner_cylinder (bool): If False, skip axon postprocessing and
            return None as the second element. Mirrors the extension's
            predict_inner_cylinder flag. Named after the InnerCylinder level in
            the extension hierarchy (Fibre > InnerCylinder > Axon), which is
            populated from the axon SDT predictions.

    Returns:
        labels_fibre (np.ndarray): Fibre instance labels.
        labels_axon (np.ndarray | None): Axon instance labels mapped to their
            parent fibre (each axon label ID equals its parent fibre label ID).
            None if predict_inner_cylinder is False.
        semantic (np.ndarray): Semantic segmentation map
            (0=background, 1=fibre, 2=inner_cylinder).
    """
    model.eval()
    model.to(device)
    # Axons are roughly half the diameter of fibres
    min_axon_diameter = min_diameter / 2

    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        raw_output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )

    prediction = raw_output.squeeze(0).cpu()  # [C, H, W]

    # Semantic segmentation (channels 0–2: bg / fibre / inner_cylinder)
    semantic = apply_semantic_segmentation_head(prediction[0:3].unsqueeze(0))[0].numpy().astype(np.uint8)

    # Distance maps (channel 3: fibre SDT, channel 4: axon SDT)
    dt_fibre = prediction[3]
    dt_axon = prediction[4]

    # Fibre instance segmentation — restrict watershed to fibre + inner cylinder mask.
    # semantic >= 1 includes both myelin (class 1) and inner cylinder (class 2) so
    # the watershed expands across the full fibre disk, not just the myelin ring.
    fibre_valid = (semantic >= 1)
    labels_fibre = segment_instances_from_sdt(
        distancemap=dt_fibre,
        threshold=fibre_threshold,
        min_diameter=min_diameter,
        valid_mask=fibre_valid,
    )
    min_fibre_area = int(np.pi * (min_diameter / 2) ** 2)
    labels_fibre = remove_small_objects(labels_fibre, max_size=max(0, min_fibre_area - 1))

    if not predict_inner_cylinder:
        return labels_fibre, None, semantic

    # Axon instance segmentation — no explicit valid_mask: segment_instances_from_sdt
    # defaults to distancemap >= 0, which correctly restricts watershed to pixels where
    # the model predicts positive axon SDT (inside the axon). Using semantic >= 2 instead
    # would flood the full inner cylinder, not the axon itself.
    labels_axon = segment_instances_from_sdt(
        distancemap=dt_axon,
        threshold=axon_threshold,
        min_diameter=min_axon_diameter,
    )
    min_axon_area = int(np.pi * (min_axon_diameter / 2) ** 2)
    labels_axon = remove_small_objects(labels_axon, max_size=max(0, min_axon_area - 1))

    labels_axon = map_axon_labels_to_fibres(labels_fibre, labels_axon)

    return labels_fibre, labels_axon, semantic
