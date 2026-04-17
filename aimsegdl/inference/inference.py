import torch
import os
import importlib.resources
from pathlib import Path
import numpy as np
from typing import Tuple
from skimage.io import imread
from skimage.morphology import remove_small_objects
from monai.inferers import sliding_window_inference
from aimsegdl.utils.model_building import model_fn
from aimsegdl.utils.image_processing import (
    normalize,
    segment_instances_from_sdt,
    apply_semantic_segmentation_head,
    map_axon_labels_to_fibres,
)


def load_image(img_path: str) -> np.ndarray:
    """Load and normalize a grayscale image."""
    image = normalize(imread(img_path).astype(np.float32))
    return image


def load_torchscript_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.jit.ScriptModule:
    """Load a TorchScript (.pt) model."""
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def resolve_model_path(model_identifier: str) -> str:
    """
    Resolve a model path for inference.

    - First, try loading a built-in pretrained weight from `aimsegdl.weights`.
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
        ref = importlib.resources.files("aimsegdl.weights").joinpath(model_identifier + ".pth")
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
    Load a PyTorch model from a resolved path (supports built-in weight names).

    Args:
        model_path (str): Path or name of the model weights (.pth).
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model in eval mode.
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
    fibre_threshold: float=0.5,
    axon_threshold: float=0.5,
    min_diameter: float=30.0,
    sw_batch_size=1,
    overlap=0.5,
    predict_inner_tongue: bool=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inference using a PyTorch model (from .pth) with MONAI's sliding window inference.

    Args:
        image (np.ndarray): Input image as a 2D array.
        model (torch.nn.Module): Loaded PyTorch model (not scripted).
        device (str): Device to run inference on.
        roi_size (tuple): Sliding window size.
        sw_batch_size (int): Sliding window batch size.
        overlap (float): Overlap between windows.
        predict_inner_tongue (bool): If False, skip axon/inner-cylinder postprocessing.

    Returns:
        labels_fibre (np.ndarray): Fibre instance labels.
        labels_axon (np.ndarray): Axon instance labels, or None if predict_inner_tongue=False.
        semantic (np.ndarray): Semantic segmentation map.
    """
    model.eval()
    model.to(device)
    min_axon_diameter = min_diameter / 2

    # Prepare input
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

    with torch.no_grad():
        # Raw model output: [1, C, H, W]
        raw_output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian"  # Optional: use 'constant' or 'gaussian'
        )

    # Post-processing
    prediction = raw_output.squeeze(0).cpu()  # shape: [C, H, W]
    
    # Semantic map
    semantic = apply_semantic_segmentation_head(prediction[0:3, :, :].unsqueeze(0))[0].numpy().astype(np.uint8)

    # Distance maps
    dt_fibre = prediction[3, :, :]
    dt_axon = prediction[4, :, :]

    # Instance segmentation
    labels_fibre = segment_instances_from_sdt(distancemap=dt_fibre, threshold=fibre_threshold, min_diameter=min_diameter, valid_mask=None, seed_mask=None)

    if not predict_inner_tongue:
        return labels_fibre, None, semantic

    labels_axon = segment_instances_from_sdt(dt_axon, axon_threshold, min_axon_diameter)
    mapped_axons = map_axon_labels_to_fibres(labels_fibre, labels_axon)

    return labels_fibre, mapped_axons, semantic
