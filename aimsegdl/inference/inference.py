import torch
import numpy as np
from typing import Tuple
from skimage.io import imread
from monai.inferers import sliding_window_inference
from aimsegdl.utils.model_building import model_fn
from aimsegdl.utils.image_processing import (
    normalize,
    segment_instances_from_sdt,
    apply_semantic_segmentation_head,
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


def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.nn.Module:
    """
    Load a PyTorch model from a .pth file.

    Args:
        model_path (str): Path to the .pth file containing model weights.
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    # Instantiate model
    model = model_fn(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(
    image: np.ndarray,
    model: torch.nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    roi_size=(512, 512),
    sw_batch_size=1,
    overlap=0.5
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

    Returns:
        labels_fibre (np.ndarray): Fibre instance labels.
        labels_axon (np.ndarray): Axon instance labels.
        semantic (np.ndarray): Semantic segmentation map.
    """
    model.eval()
    model.to(device)

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

    labels_fibre = segment_instances_from_sdt(dt_fibre)
    labels_axon = segment_instances_from_sdt(dt_axon)

    return labels_fibre, labels_axon, semantic
