import torch
import torch.nn as nn
import torch.nn.functional as F
from aimsegdl.utils.image_processing import apply_semantic_segmentation_head_scriptable


class Pipeline(nn.Module):
    """
    TorchScript-compatible inference pipeline.

    This wraps a model and handles input padding, semantic head application,
    and output reassembly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.target_height = 512
        self.target_width = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with padding, inference, semantic head, and unpadding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor after postprocessing (1, C_out, H, W)
        """
        b, c, h, w = x.shape
        pad_h = max(0, self.target_height - h)
        pad_w = max(0, self.target_width - w)

        # Pad input
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant')

        # Model forward
        pred = self.model(x)

        # Semantic segmentation postprocessing
        semantic = apply_semantic_segmentation_head_scriptable(pred[:, 0:3, :, :])
        dt_fibre = pred[:, 3, :, :]
        dt_axon = pred[:, 4, :, :]

        # Combine all outputs
        output = torch.cat((semantic, dt_fibre, dt_axon), dim=0).unsqueeze(0)

        # Remove padding
        output = output[:, :, :h, :w]

        return output.float()