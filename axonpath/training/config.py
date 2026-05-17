from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    """Configuration for model training.

    All paths are relative to the working directory unless absolute paths are given.
    """
    learning_rate: float = 1e-3
    batch_size: int = 8             # < 8 uses instance norm; >= 8 uses batch norm
    num_epochs: int = 1000
    # num_workers=0 is safe on all platforms.
    # On Linux, increasing is low-cost (fork/copy-on-write).
    # On Windows, each worker spawns a full process and copies the dataset cache —
    # RAM scales with num_workers × cache size. Increase with caution.
    num_workers: int = 0
    image_height: int = 512
    image_width: int = 512
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fibre_threshold: float = 0.5
    axon_threshold: float = 0.5
    min_diameter: float = 30.0      # pixels; axon minimum is derived as min_diameter / 2
    train_dir: str = "prepared_data/train_tiles/"
    val_dir: str = "prepared_data/val_tiles/"
    pretrained_weights: str = None  # path to a .pth file or built-in weight name (e.g. 'axonpath')
    load_checkpoint: bool = False   # resume training from <output_dir>/model_checkpoint.pth.tar
    model_name: str = "axonpath"
    output_dir: str = "."           # all training outputs (weights, checkpoint, plots) are written here
    seed: int = None                # set for reproducible runs; None = non-deterministic
    use_lr_scheduler: bool = True   # ReduceLROnPlateau on val_loss (factor=0.5, patience=50)
    lr_scheduler_patience: int = 50
    loss_weights: tuple = (1.0, 1.0, 1.0)  # weights for (CE, MSE_fibre, MSE_axon) loss terms
    ce_weight_ic: float = 1.0              # CE class weight for inner_cylinder (class 2); bg fixed at 1.0
    ce_weight_myelin: float = 1.0          # CE class weight for myelin (class 1); bg fixed at 1.0
    early_stopping_patience: int = 0        # 0 = disabled; stop if balanced F1 does not improve
    early_stopping_min_delta: float = 0.001 # minimum improvement to reset the patience counter
