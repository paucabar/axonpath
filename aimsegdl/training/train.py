import torch
import torch.nn as nn
import torch.optim as optim
from aimsegdl.transforms.custom_transforms import transforms_fn
from aimsegdl.utils import (
    model_fn, get_datasets, get_loaders, load_checkpoint, save_checkpoint,
    evaluate_fn, loss_plot_fn, loss_plot_log_fn, plot_segmentation_scores_fn
)
from aimsegdl.training.train_loop import train_loop

def train(hparams: dict):
    device = hparams["device"]

    train_tf, val_tf = transforms_fn(hparams["image_height"], hparams["image_width"])
    train_ds, val_ds = get_datasets(hparams["train_dir"], hparams["val_dir"], train_tf, val_tf)
    train_ds.populate_cache()
    val_ds.populate_cache()

    norm_type = "group" if hparams["batch_size"] < 8 else "batch"
    model = model_fn(device, norm_type=norm_type)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loss_fns = [ce_loss, mse_loss]
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    scaler = torch.cuda.amp.GradScaler()

    train_loss, val_loss = [], []
    f1_fibre, f1_axon, dice_score = [], [], []
    balanced_seg_score, best_score = [], 0
    last_epoch = 0

    if hparams["load_model"]:
        last_epoch, train_loss, val_loss, f1_fibre, f1_axon, dice_score, balanced_seg_score, best_score = \
            load_checkpoint(torch.load("model_checkpoint.pth.tar"), model, optimizer)

    for epoch in range(hparams["epochs"]):
        print(f"\nEpoch {epoch + 1}/{hparams['epochs']}")
        if hparams["load_model"]:
            print(f"Total epoch {last_epoch + epoch + 1}/{last_epoch + hparams['epochs']}")

        train_loader, val_loader = get_loaders(train_ds, val_ds, hparams["batch_size"], hparams["num_workers"], hparams["pin_memory"])

        t_loss = train_loop(train_loader, model, optimizer, loss_fns, scaler, device)
        train_loss.append(t_loss)

        show_results = hparams["show_val_interval"] > 0 and epoch % hparams["show_val_interval"] == 0
        v_loss, f1_fib, f1_ax, dice = evaluate_fn(val_loader, model, loss_fns, device, show_results)
        val_loss.append(v_loss)
        f1_fibre.append(f1_fib)
        f1_axon.append(f1_ax)
        dice_score.append(dice)

        score = 0.4 * f1_fib + 0.4 * f1_ax + 0.2 * dice
        balanced_seg_score.append(score)

        print(f"Train: {t_loss:.4f} | Val: {v_loss:.4f} | F1: {f1_fib:.4f}/{f1_ax:.4f} | Dice: {dice:.4f} | BSS: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_weights_model.pth")

        save_checkpoint({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1 + last_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_fibre": f1_fibre,
            "f1_axon": f1_axon,
            "dice_score": dice_score,
            "balanced_segmentation_score": balanced_seg_score,
            "best_score": best_score,
        })

    torch.save(model.state_dict(), "last_epoch_model.pth")
    loss_plot_fn(train_loss, val_loss)
    loss_plot_log_fn(train_loss, val_loss)
    plot_segmentation_scores_fn(f1_fibre, f1_axon, dice_score, balanced_seg_score)
