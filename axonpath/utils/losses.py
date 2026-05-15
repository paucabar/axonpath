from monai.losses import DiceCELoss

_dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)

def compute_loss(pred, targets, loss_fns, loss_weights=(1.0, 1.0, 1.0)):
    w_ce, w_mse_fibre, w_mse_axon = loss_weights
    sem_loss = w_ce * _dice_ce(pred[:, 0:3, :, :], targets[:, 2:3, :, :].long())
    mse1 = w_mse_fibre * loss_fns[0](pred[:, 3, :, :], targets[:, 3, :, :].float())
    mse2 = w_mse_axon * loss_fns[0](pred[:, 4, :, :], targets[:, 4, :, :].float())
    return sem_loss + mse1 + mse2
