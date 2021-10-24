import os

import torch
import torch.nn.functional as F

Laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).cuda()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork1':
        # 400ep
        return lambda epoch: -epoch / 200 + 2.5 if epoch > 300 else 1
    elif lr_lambda_tag == 'subnetwork2':
        # 400ep
        # return lambda epoch: 1
        return lambda epoch: -epoch / 200 + 2.5 if epoch > 300 else 1
    elif lr_lambda_tag == 'full':
        # 300ep
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    pad = [1, 1, 1, 1]
    laplacian_kernel = Laplacian.view(1, 1, 3, 3)
    edge_tensor = torch.zeros(img_tensor.shape, dtype=torch.float32).cuda()
    for i in range(img_tensor.shape[1]):
        padded = F.pad(img_tensor[:, i:i + 1, :, :], pad, mode='reflect')
        edge_tensor[:, i:i + 1, :, :] = F.conv2d(padded, laplacian_kernel)
    return edge_tensor
