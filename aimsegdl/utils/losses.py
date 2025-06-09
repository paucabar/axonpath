def compute_loss(pred, targets, loss_fns):
    ce_loss = loss_fns[0](pred[:, 0:3, :, :], targets[:, 2, :, :].long())
    mse1 = loss_fns[1](pred[:, 3, :, :], targets[:, 3, :, :].float())
    mse2 = loss_fns[1](pred[:, 4, :, :], targets[:, 4, :, :].float())
    return ce_loss + mse1 + mse2
