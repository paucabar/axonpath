from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 8
    num_epochs: int = 20
    num_workers: int = 0
    image_height: int = 512
    image_width: int = 512
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir: str = "prepared_data_em/train_tiles/"
    val_dir: str = "prepared_data_em/val_tiles/"
    load_checkpoint: bool = False       # Resume from checkpoint (hidden feature)
    pretrained_weights: str = ""        # .pt weights (preferred entry point)
    bioimageio: bool = False
    model_name: str = "aimsegdl"
