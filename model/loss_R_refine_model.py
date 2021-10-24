import torch.nn.functional as F


def l1_and_l2(R_pred, R_gt, **kwargs):
    R_l1_loss_lambda = kwargs.get('R_l1_loss_lambda', 1)
    R_l1_loss = F.l1_loss(R_pred, R_gt) * R_l1_loss_lambda

    R_l2_loss_lambda = kwargs.get('R_l2_loss_lambda', 1)
    R_l2_loss = F.mse_loss(R_pred, R_gt) * R_l2_loss_lambda

    print('R_l1_loss:', R_l1_loss.item())
    print('R_l2_loss:', R_l2_loss.item())
    return R_l1_loss + R_l2_loss
