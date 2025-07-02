from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 8
    num_epochs: int = 1000
    num_workers: int = 0
    image_height: int = 512
    image_width: int = 512
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fibre_threshold: float=0.5
    axon_threshold: float=0.5
    min_diameter: float=30.0
    train_dir: str = "prepared_data/train_tiles/"
    val_dir: str = "prepared_data/val_tiles/"
    pretrained_weights: str = None      # Accepts either a path or model name (e.g., 'aimsegdl')
    load_checkpoint: bool = False       # Resume from checkpoint (hidden feature)
    bioimageio: bool = False
    model_name: str = "aimsegdl"
