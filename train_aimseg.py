import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from aimsegdl.transforms.custom_transforms import transforms_fn
from aimsegdl.utils import (
    load_checkpoint,
    save_checkpoint,
    model_fn,
    get_datasets,
    get_loaders,
    evaluate_fn,
    loss_plot_fn,
    loss_plot_log_fn,
    plot_segmentation_scores_fn,
)

# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 0
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "prepared_data_em/train_tiles/"
VAL_IMG_DIR = "prepared_data_em/val_tiles/"
SHOW_VAL_INTERVAL = 20
BIOIMAGEIO = False
MODEL_NAME = "aimsegdl"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    train_loss_all = []

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = np.stack(targets, axis=1)
        targets = torch.tensor(targets).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data).to(torch.float)
            train_crossentropy_loss = loss_fn[0](predictions[:, 0:3, :, :], targets[:, 2, :, :].long())
            train_mse_loss1 = loss_fn[1](predictions[:, 3, :, :], targets[:, 3, :, :].float())
            train_mse_loss2 = loss_fn[1](predictions[:, 4, :, :], targets[:, 4, :, :].float())
            train_loss = train_crossentropy_loss + train_mse_loss1 + train_mse_loss2

        optimizer.zero_grad()
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(train_loss=train_loss.item())
        train_loss_all.append(train_loss.item())

    return sum(train_loss_all) / len(train_loss_all)

def main():
    # Check GPU
    print("CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU available")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory}")
    else:
        print("No GPU found.")

    train_transform, val_transform = transforms_fn(IMAGE_HEIGHT, IMAGE_WIDTH)
    train_dataset, val_dataset = get_datasets(TRAIN_IMG_DIR, VAL_IMG_DIR, train_transform, val_transform)

    # Load image data in train and val dataset caches
    train_dataset.populate_cache()
    print(train_dataset)
    val_dataset.populate_cache()
    print(val_dataset)

    norm_type = "group" if BATCH_SIZE < 8 else "batch"
    model = model_fn(DEVICE, norm_type=norm_type)

    crossentropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loss_fn = [crossentropy_loss, mse_loss]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loss, val_loss, f1_fibre, f1_axon, dice_score = [], [], [], [], []
    balanced_segmentation_score = []
    best_score = 0
    last_epoch = 0

    if LOAD_MODEL:
        last_epoch, train_loss, val_loss, f1_fibre, f1_axon, dice_score, balanced_segmentation_score, best_score = \
            load_checkpoint(torch.load("model_checkpoint.pth.tar"), model, optimizer)

        train_loader, val_loader = get_loaders(train_dataset, val_dataset, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
        val_loss_load, f1_fibre_load, f1_axon_load, dice_score_load = evaluate_fn(val_loader, model, loss_fn, device=DEVICE, show_results=True)
        print(f"Model loaded. Val loss: {val_loss_load:.4f}, Fibre F1: {f1_fibre_load:.4f}, Axon F1: {f1_axon_load:.4f}, Dice: {dice_score_load:.4f}")
    else:
        print("Training model from scratch")

    display_check = np.arange(SHOW_VAL_INTERVAL - 1, NUM_EPOCHS, SHOW_VAL_INTERVAL) if SHOW_VAL_INTERVAL > 0 else 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        if LOAD_MODEL:
            print(f"Total epoch {last_epoch + epoch + 1}/{last_epoch + NUM_EPOCHS}")

        train_loader, val_loader = get_loaders(train_dataset, val_dataset, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

        t_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_loss.append(t_loss)

        show_results = np.any(display_check == epoch)
        v_loss, f1_fib, f1_ax, dice = evaluate_fn(val_loader, model, loss_fn, device=DEVICE, show_results=show_results)
        val_loss.append(v_loss)
        f1_fibre.append(f1_fib)
        f1_axon.append(f1_ax)
        dice_score.append(dice)

        base_score = 0.4 * f1_fib + 0.4 * f1_ax + 0.2 * dice
        balanced_segmentation_score.append(base_score)

        print(f"Train loss: {t_loss:.4f} | Val loss: {v_loss:.4f} | F1 fibre: {f1_fib:.4f} | F1 axon: {f1_ax:.4f} | Dice score: {dice:.4f} | BSS: {base_score:.4f}")

        if base_score > best_score:
            best_score = base_score
            torch.save(model.state_dict(), "best_weights_model.pth")
            print("Saving best weights model")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1 + last_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_fibre": f1_fibre,
            "f1_axon": f1_axon,
            "dice_score": dice_score,
            "balanced_segmentation_score": balanced_segmentation_score,
            "best_score": best_score,
        }
        save_checkpoint(checkpoint)

    torch.save(model.state_dict(), "last_epoch_model.pth")
    loss_plot_fn(train_loss, val_loss)
    loss_plot_log_fn(train_loss, val_loss)
    plot_segmentation_scores_fn(f1_fibre, f1_axon, dice_score, balanced_segmentation_score)

    # Export model
    export_model = model_fn(DEVICE)
    export_model.load_state_dict(torch.load("best_weights_model.pth"))
    if BIOIMAGEIO:
        print("Exporting model to BioImage.IO format...")
        from aimsegdl.export_utils.model_export import export_bioimageio
        export_model.load_state_dict(torch.load("best_weights_model.pth"))
        export_bioimageio(export_model, MODEL_NAME + "_bioimageio", True, r"data_tem/test_images/P03B_Frame6_t0.tif")
    else:
        print("Exporting torchscript...")
        from aimsegdl.export_utils.model_export import export_torchscript_model
        export_torchscript_model(export_model, MODEL_NAME + ".pt")


if __name__ == "__main__":
    main()
