import argparse
import torch
from aimsegdl.training.config import TrainingConfig
from aimsegdl.training.train import train
from aimsegdl.data_preparation.data_preparation import create_train_val_test_split_all


def main():
    parser = argparse.ArgumentParser(description="Train AimSegDL model")

    # Optional overrides for key parameters
    parser.add_argument("--data_input", type=str, help="Input folder with raw datasets (default: datasets/)")
    parser.add_argument("--data_output", type=str, help="Output folder for prepared tiles (default: prepared_data/)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--image_height", type=int, help="Image height")
    parser.add_argument("--image_width", type=int, help="Image width")
    parser.add_argument("--pretrained_weights", type=str, help="Path to .pt weights file to initialize model")
    parser.add_argument("--load_checkpoint", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--bioimageio", action="store_true", help="Export BioImage.IO package")
    parser.add_argument("--model_name", type=str, help="Name for saving model and logs")

    args = parser.parse_args()

    # Data preparation ---
    input_root = args.data_input or "datasets"
    output_root = args.data_output or "prepared_data"
    create_train_val_test_split_all(in_root=input_root, out_root=output_root)

    # Define training configuration
    config = TrainingConfig(
        learning_rate=args.learning_rate if args.learning_rate is not None else TrainingConfig.learning_rate,
        batch_size=args.batch_size if args.batch_size is not None else TrainingConfig.batch_size,
        num_epochs=args.num_epochs if args.num_epochs is not None else TrainingConfig.num_epochs,
        num_workers=args.num_workers if args.num_workers is not None else TrainingConfig.num_workers,
        image_height=args.image_height if args.image_height is not None else TrainingConfig.image_height,
        image_width=args.image_width if args.image_width is not None else TrainingConfig.image_width,
        pin_memory=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_dir=f"{output_root}/train_tiles/",
        val_dir=f"{output_root}/val_tiles/",
        pretrained_weights=args.pretrained_weights,
        load_checkpoint=args.load_checkpoint,
        bioimageio=args.bioimageio,
        model_name=args.model_name if args.model_name else TrainingConfig.model_name,
    )

    # Train
    train(config)


if __name__ == "__main__":
    main()
