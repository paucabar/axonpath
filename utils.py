
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import monai

from dataset import AimSegDataset

from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
import matplotlib as mpl
import numpy as np
import torch as t
import matplotlib.pyplot as plt

from image_processing import (
    postprocessing_distmap,
    postprocessing_distmap_invedge,
    postprocessing_signed_map,
    postprocessing_sdt,
    last_layer_fn,
)

def show_images(*img_list,binaries=[],titles=[],save_str=False,n_cols=3,axes=False,cmap="plasma",labels=[],dpi=None):
    """Designed to plot torch tensor and numpy arrays in windows robustly"""    
    if dpi:
        mpl.rcParams['figure.dpi'] = dpi    
        img_list = [img for img in img_list]
    if isinstance(img_list[0], list):
        img_list = img_list[0]
    rows=(len(img_list)-1)//n_cols+1    
    columns=np.min([n_cols,len(img_list)])
    fig = plt.figure(figsize=(5*(columns+1),5*(rows+1)))
    fig.tight_layout() 
    grid = plt.GridSpec(rows,columns,figure=fig)
    grid.update(wspace=0.2, hspace=0, left = None, right =None, bottom = None, top = None)
    for i,img in enumerate(img_list):
        if t.is_tensor(img):
            img=t.squeeze(img).detach().cpu().numpy()
        if len(img.shape)>2:
            img=np.moveaxis(img,np.argmin(img.shape),-1)
            if img.shape[-1]>3 or img.shape[-1]==2:
                show_images([img[...,i] for i in range(img.shape[-1])],binaries=binaries,titles=["Channel:"+str(i) for i in range(img.shape[-1])],save_str=save_str,n_cols=n_cols,axes=axes,cmap=cmap)
                continue        
        ax1 = plt.subplot(grid[i])
        if not axes:
            plt.axis('off')
        if i in binaries:
            im=ax1.imshow(img,vmin=0,vmax=1,cmap=cmap,interpolation='nearest')
        if i in labels:
            l=cc.cm.glasbey_bw_minc_20_minl_30_r.colors            
            l[0]=[0,0,0]
            cmap_lab = LinearSegmentedColormap.from_list('my_list', l, N=1000)
            im=ax1.imshow(img,cmap=cmap_lab,interpolation='nearest')
        else:
            im=ax1.imshow(img,cmap=cmap)
        plt.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)
        if len(titles)==len(img_list):
            ax1.set_title(titles[i])
    if not save_str:
        plt.show()
    if save_str:
        plt.savefig(save_str+".png",bbox_inches='tight')
        plt.close()
        return None

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss'] 
    val_loss = checkpoint['val_loss']
    print("Loading checkpoint")
    return last_epoch, train_loss, val_loss

def model_fn(device):
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),#(16, 32, 64, 128, 256) or (32, 64, 128, 256, 512)
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.25,
    ).to(device)
    return model

def get_datasets(
        train_dir,
        train_maskdir,
        train_labeldir,
        val_dir,
        val_maskdir,
        val_labeldir,
        train_transform,
        val_transform,
    ):
    train_dataset = AimSegDataset(
        image_dir=train_dir,
        masksem_dir=train_maskdir,
        maskins_dir=train_labeldir,
        transform=train_transform,
    )
    val_dataset = AimSegDataset(
        image_dir=val_dir,
        masksem_dir=val_maskdir,
        maskins_dir=val_labeldir,
        transform=val_transform,
    )
    return train_dataset, val_dataset

def get_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def accuracy_fn(sem_pred, sem_targ):
    intersection = (sem_pred == sem_targ).sum()
    pixels_total = torch.numel(sem_pred)
    accuracy = intersection / pixels_total
    return accuracy

def evaluate_fn(loader, model, loss_fn, device="cuda", show_results=False):
    model.eval()
    val_loss = []

    with torch.no_grad():
        for x, y in loader:
            # data to device
            x = x.to(device)
            # predict
            prediction = model(x)
            # targets to device
            y = np.stack(y, axis=1)
            y = torch.tensor(y).to(device=device)
            # get loss functions
            val_crossentropy_loss = loss_fn[0](prediction[:, 0:3, :, :], y[:, 1, :, :].long())#.item()
            val_mse_loss1 = loss_fn[1](prediction[:, 3, :, :], y[:, 2, :, :].float())#.item()
            val_mse_loss2 = loss_fn[1](prediction[:, 4, :, :], y[:, 3, :, :].float())
            val_crossentropy_loss = val_crossentropy_loss.float()
            val_mse_loss1 = val_mse_loss1.float()
            val_mse_loss2 = val_mse_loss2.float()
            val_mse_loss = val_mse_loss1 + val_mse_loss2
            val_loss.append(torch.add(val_crossentropy_loss, val_mse_loss).item())

        val_loss_mean = sum(val_loss)/len(val_loss)

        if (show_results):
            pred = model(x[0:1])
            semantic = last_layer_fn(pred[0:1, 0:3, :, :])
            labels_fibre = postprocessing_sdt(pred[0:1, 3, :, :])
            labels_axon = postprocessing_sdt(pred[0:1, 4, :, :])
            print(pred.shape)
            show_images(
                x[0],
                y[0, 1, :, :],
                y[0, 0, :, :],
                pred[0, 3, :, :],
                pred[0, 4, :, :],
                semantic,
                labels_fibre,
                labels_axon,
                titles = ["Image", "Target Semantic", "Target Instance", "Pred Fibre Distance Transform", "Pred Axon Distance Transform", "Prediction Semantic", "Prediction Fibre Instance", "Prediction Axon Instance"],
                n_cols = 4
                )
    model.train()
    
    return val_loss_mean

def loss_plot_fn(train_loss, val_loss):
    # plot train and val loss
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss")
    
    fig_loss = plt.gcf()
    
    plt.plot(np.arange(1,len(train_loss)+1).tolist(), train_loss, label = "Training loss")
    plt.plot(np.arange(1,len(val_loss)+1).tolist(), val_loss, label = "Validation loss")
    plt.title('Training and validation loss vs epoch number (linear)')
    plt.ylabel("Loss")
    #plt.ylim(0,0.2)
    plt.xlabel("Epoch number")
    plt.xticks(ticks=np.arange(0, len(val_loss)+1, (len(val_loss))/4).tolist())
    plt.legend()
    plt.show()
    
    fig_loss.set_facecolor('white')
    fig_loss.savefig('loss_plot.png', bbox_inches='tight', dpi=300)

def loss_plot_log_fn(train_loss, val_loss):
    # Plot train and val loss in a log scale
    print("\n----------------------------------------------------------------------------")
    print("\nTraining and validation loss")
    
    fig_loss = plt.gcf()
    
    plt.plot(np.arange(1, len(train_loss) + 1).tolist(), train_loss, label="Training loss")
    plt.plot(np.arange(1, len(val_loss) + 1).tolist(), val_loss, label="Validation loss")
    
    plt.yscale('log')  # Apply a logarithmic scale to the y-axis
    plt.title('Training and validation loss vs epoch number (log)')
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch number")
    plt.xticks(ticks=np.arange(0, len(val_loss)+1, (len(val_loss))/4).tolist())
    plt.legend()    
    plt.show()
    
    fig_loss.set_facecolor('white')
    fig_loss.savefig('loss_plot_log.png', bbox_inches='tight', dpi=300)

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()



