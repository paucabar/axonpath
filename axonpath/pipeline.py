import torch
import torch.nn as nn
import torch.nn.functional as F
from axonpath.inference.post_processing import apply_semantic_segmentation_head_scriptable

def pad_to_multiple(x: int, multiple: int) -> int:
    return (multiple - x % multiple) % multiple

class Pipeline(nn.Module):
    """
    TorchScript-compatible inference pipeline.

    This wraps a model and handles input padding, semantic head application,
    and output reassembly.
    """

    def __init__(self, model: nn.Module, target_height: int = 512, target_width: int = 512):
        super().__init__()
        self.model = model
        self.target_height = target_height
        self.target_width = target_width


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, orig_h, orig_w = x.shape  # Save original input size

        # Enforce minimum size (e.g. 512) before padding to multiple of 32
        padded_h = max(orig_h, self.target_height)
        padded_w = max(orig_w, self.target_width)

        # Compute padding to reach next multiple of 32
        pad_h = pad_to_multiple(padded_h, 32)
        pad_w = pad_to_multiple(padded_w, 32)

        final_h = padded_h + pad_h
        final_w = padded_w + pad_w

        # Pad image (right and bottom only)
        x = F.pad(x, (0, final_w - orig_w, 0, final_h - orig_h), mode='constant')  # [B, C, H_pad, W_pad]

        # Forward through model
        pred = self.model(x)  # [B, 5, H_pad, W_pad]

        # Apply semantic segmentation head
        semantic = apply_semantic_segmentation_head_scriptable(pred[:, 0:3, :, :]).unsqueeze(1).to(dtype=pred.dtype)
        dt_fibre = pred[:, 3:4, :, :]
        dt_axon = pred[:, 4:5, :, :]

        # Combine outputs
        output = torch.cat([semantic, dt_fibre, dt_axon], dim=1)

        # Crop to original input size
        output = output[:, :, :orig_h, :orig_w]

        return output.float()



