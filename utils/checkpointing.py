import torch


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss'] 
    val_loss = checkpoint['val_loss']
    f1_fibre = checkpoint['f1_fibre']
    f1_axon = checkpoint['f1_axon']
    dice_score = checkpoint['dice_score']
    balanced_segmentation_score = checkpoint['balanced_segmentation_score']
    best_score = checkpoint['best_score']
    print("Loading checkpoint")
    return last_epoch, train_loss, val_loss, f1_fibre, f1_axon, dice_score, balanced_segmentation_score, best_score