import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class _WeightedDiceCE(nn.Module):
    """DiceLoss + class-weighted CrossEntropyLoss for semantic segmentation.

    Equivalent to MONAI's DiceCELoss(to_onehot_y=True, softmax=True) but with
    per-class CE weights, which DiceCELoss did not support until MONAI 1.6.
    """

    def __init__(self, ce_weight: torch.Tensor):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        # register_buffer so .to(device) moves the weight with the module
        self.register_buffer("ce_weight", ce_weight)

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits; target: (B, 1, H, W) class indices
        dice_loss = self.dice(pred, target)
        ce_loss = F.cross_entropy(pred, target.squeeze(1), weight=self.ce_weight)
        return dice_loss + ce_loss


def make_dice_ce(ce_weight_ic: float = 1.0, ce_weight_myelin: float = 1.0, device: str = "cpu") -> _WeightedDiceCE:
    """Create a DiceCE loss with optional per-class CE weights for myelin (class 1) and inner_cylinder (class 2)."""
    ce_weight = torch.tensor([1.0, ce_weight_myelin, ce_weight_ic], device=device)
    return _WeightedDiceCE(ce_weight)


def compute_loss(pred, targets, loss_fns, loss_weights=(1.0, 1.0, 1.0)):
    # loss_fns[0]: MSELoss for fibre/axon SDT; loss_fns[1]: _WeightedDiceCE for semantic
    w_ce, w_mse_fibre, w_mse_axon = loss_weights
    sem_loss = w_ce * loss_fns[1](pred[:, 0:3, :, :], targets[:, 2:3, :, :].long())
    mse1 = w_mse_fibre * loss_fns[0](pred[:, 3, :, :], targets[:, 3, :, :].float())
    mse2 = w_mse_axon * loss_fns[0](pred[:, 4, :, :], targets[:, 4, :, :].float())
    return sem_loss + mse1 + mse2
