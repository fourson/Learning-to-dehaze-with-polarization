import torch.nn.functional as F


def l1_and_l2(P_A_pred, P_A_gt, P_T_pred, P_T_gt, T_pred, T_gt, **kwargs):
    P_A_l1_loss_lambda = kwargs.get('P_A_l1_loss_lambda', 1)
    P_A_l1_loss = F.l1_loss(P_A_pred, P_A_gt) * P_A_l1_loss_lambda

    P_A_l2_loss_lambda = kwargs.get('P_A_l2_loss_lambda', 1)
    P_A_l2_loss = F.mse_loss(P_A_pred, P_A_gt) * P_A_l2_loss_lambda

    P_T_l1_loss_lambda = kwargs.get('P_T_l1_loss_lambda', 1)
    P_T_l1_loss = F.l1_loss(P_T_pred, P_T_gt) * P_T_l1_loss_lambda

    P_T_l2_loss_lambda = kwargs.get('P_T_l2_loss_lambda', 1)
    P_T_l2_loss = F.mse_loss(P_T_pred, P_T_gt) * P_T_l2_loss_lambda

    T_l1_loss_lambda = kwargs.get('T_l1_loss_lambda', 1)
    T_l1_loss = F.l1_loss(T_pred, T_gt) * T_l1_loss_lambda

    T_l2_loss_lambda = kwargs.get('T_l2_loss_lambda', 1)
    T_l2_loss = F.mse_loss(T_pred, T_gt) * T_l2_loss_lambda

    print('P_A_l1_loss:', P_A_l1_loss.item())
    print('P_A_l2_loss:', P_A_l2_loss.item())
    print('P_T_l1_loss:', P_T_l1_loss.item())
    print('P_T_l2_loss:', P_T_l2_loss.item())
    print('T_l1_loss:', T_l1_loss.item())
    print('T_l2_loss:', T_l2_loss.item())

    return P_A_l1_loss + P_A_l2_loss + P_T_l1_loss + P_T_l2_loss + T_l1_loss + T_l2_loss

