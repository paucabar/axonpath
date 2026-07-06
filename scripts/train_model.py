import argparse
import torch
from axonpath.training.config import TrainingConfig
from axonpath.training.train import train

def main():
    parser = argparse.ArgumentParser(description="Train axonpath model")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=TrainingConfig.learning_rate, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=TrainingConfig.num_epochs, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=TrainingConfig.num_workers, help="Number of data loading workers")
    parser.add_argument("--image_height", type=int, default=TrainingConfig.image_height, help="Image height")
    parser.add_argument("--image_width", type=int, default=TrainingConfig.image_width, help="Image width")

    # Inference / segmentation parameters
    parser.add_argument("--fibre_threshold", type=float, default=TrainingConfig.fibre_threshold, help="Fibre threshold for SDT")
    parser.add_argument("--axon_threshold", type=float, default=TrainingConfig.axon_threshold, help="Axon threshold for SDT")
    parser.add_argument("--min_diameter", type=float, default=TrainingConfig.min_diameter, help="Minimum object diameter")

    # Model init / export
    parser.add_argument("--train_dir", type=str, default=TrainingConfig.train_dir, help="Training data directory")
    parser.add_argument("--val_dir", type=str, default=TrainingConfig.val_dir, help="Validation data directory")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to .pth file or model name")
    parser.add_argument("--load_checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--model_name", type=str, default=TrainingConfig.model_name, help="Name for saving model and logs")
    parser.add_argument("--output_dir", type=str, default=TrainingConfig.output_dir, help="Directory for all training outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no_lr_scheduler", action="store_true", help="Disable ReduceLROnPlateau learning rate scheduler")
    parser.add_argument("--mse_fibre_weight", type=float, default=1.0, help="Loss weight for fibre SDT MSE term")
    parser.add_argument("--mse_axon_weight", type=float, default=1.0, help="Loss weight for axon SDT MSE term")
    parser.add_argument("--early_stopping_patience", type=int, default=TrainingConfig.early_stopping_patience, help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=TrainingConfig.early_stopping_min_delta, help="Minimum F1 improvement to reset early stopping counter")
    parser.add_argument("--lr_scheduler_patience", type=int, default=TrainingConfig.lr_scheduler_patience, help="Patience for ReduceLROnPlateau scheduler (epochs)")
    parser.add_argument("--ce_weight_ic", type=float, default=TrainingConfig.ce_weight_ic,
                        help="CE class weight for inner_cylinder (class 2); bg fixed at 1.0")
    parser.add_argument("--ce_weight_myelin", type=float, default=TrainingConfig.ce_weight_myelin,
                        help="CE class weight for myelin (class 1); bg fixed at 1.0")

    args = parser.parse_args()

    # Create training config
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        image_height=args.image_height,
        image_width=args.image_width,
        pin_memory=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fibre_threshold=args.fibre_threshold,
        axon_threshold=args.axon_threshold,
        min_diameter=args.min_diameter,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pretrained_weights=args.pretrained_weights,
        load_checkpoint=args.load_checkpoint,
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
        use_lr_scheduler=not args.no_lr_scheduler,
        loss_weights=(1.0, args.mse_fibre_weight, args.mse_axon_weight),
        ce_weight_ic=args.ce_weight_ic,
        ce_weight_myelin=args.ce_weight_myelin,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler_patience=args.lr_scheduler_patience,
    )

    # Train
    train(config)


if __name__ == "__main__":
    main()
