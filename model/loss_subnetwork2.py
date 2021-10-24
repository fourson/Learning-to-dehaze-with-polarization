import torch.nn.functional as F


def l1_and_l2(A_infinity_pred, A_infinity_gt, R_pred, R_gt, **kwargs):
    A_infinity_l1_loss_lambda = kwargs.get('A_infinity_l1_loss_lambda', 1)
    A_infinity_l1_loss = F.l1_loss(A_infinity_pred, A_infinity_gt) * A_infinity_l1_loss_lambda

    A_infinity_l2_loss_lambda = kwargs.get('A_infinity_l2_loss_lambda', 1)
    A_infinity_l2_loss = F.mse_loss(A_infinity_pred, A_infinity_gt) * A_infinity_l2_loss_lambda

    R_l1_loss_lambda = kwargs.get('R_l1_loss_lambda', 1)
    R_l1_loss = F.l1_loss(R_pred, R_gt) * R_l1_loss_lambda

    R_l2_loss_lambda = kwargs.get('R_l2_loss_lambda', 1)
    R_l2_loss = F.mse_loss(R_pred, R_gt) * R_l2_loss_lambda

    print('A_infinity_l1_loss:', A_infinity_l1_loss.item())
    print('A_infinity_l2_loss:', A_infinity_l2_loss.item())
    print('R_l1_loss:', R_l1_loss.item())
    print('R_l2_loss:', R_l2_loss.item())
    return A_infinity_l1_loss + A_infinity_l2_loss + R_l1_loss + R_l2_loss

