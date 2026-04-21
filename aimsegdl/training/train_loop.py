import torch
from tqdm import tqdm
import numpy as np
from aimsegdl.utils.losses import compute_loss

def train_loop(loader, model, optimizer, loss_fns, scaler, device, loss_weights=(1.0, 1.0, 1.0)):
    loop = tqdm(loader)
    train_loss_all = []

    for data, targets in loop:
        data = data.to(device)
        targets = torch.tensor(np.stack(targets, axis=1)).to(device)

        with torch.amp.autocast(device_type=device):
            predictions = model(data).float()
            loss = compute_loss(predictions, targets, loss_fns, loss_weights)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(train_loss=loss.item())
        train_loss_all.append(loss.item())

    return np.mean(train_loss_all)
