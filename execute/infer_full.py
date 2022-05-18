import os
import argparse
import importlib
import sys

import torch
from tqdm import tqdm
import numpy as np


def infer_default():
    if R_only:
        R_pred_dir = os.path.join(result_dir, 'R_pred')
        util.ensure_dir(R_pred_dir)
    else:
        P_A_pred_dir = os.path.join(result_dir, 'P_A_pred')
        P_T_pred_dir = os.path.join(result_dir, 'P_T_pred')
        T_pred_dir = os.path.join(result_dir, 'T_pred')
        A_infinity_pred_dir = os.path.join(result_dir, 'A_infinity_pred')
        R_pred_dir = os.path.join(result_dir, 'R_pred')
        util.ensure_dir(P_A_pred_dir)
        util.ensure_dir(P_T_pred_dir)
        util.ensure_dir(T_pred_dir)
        util.ensure_dir(A_infinity_pred_dir)
        util.ensure_dir(R_pred_dir)
        # use this to output T_hat for visualization
        # T_hat_pred_dir = os.path.join(result_dir, 'T_hat_pred')
        # R_hat_pred_dir = os.path.join(result_dir, 'R_hat_pred')
        # util.ensure_dir(T_hat_pred_dir)
        # util.ensure_dir(R_hat_pred_dir)

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0]

            # get data and send them to GPU
            I_alpha = sample['I_alpha'].to(device)
            # (N, C, H, W) GPU tensor
            I = sample['I'].to(device)
            delta_I = sample['delta_I'].to(device)

            # note that P_A_pred, P_T_pred, and A_infinity_pred are (N, C, H, W) GPU tensor, not (N, C) like original data
            P_A_pred, P_T_pred, T_pred, A_infinity_pred, R_pred = model(I_alpha, I, delta_I)
            # use this to output T_hat for visualization
            # P_A_pred, P_T_pred, T_pred, A_infinity_pred, R_pred, T_hat_pred, R_hat_pred = model(I_alpha, I, delta_I)
            if R_only:
                R_pred_numpy = np.transpose(R_pred.squeeze().cpu().numpy(), (1, 2, 0))
                np.save(os.path.join(R_pred_dir, name + '.npy'), R_pred_numpy)
            else:
                P_A_pred_numpy = np.transpose(P_A_pred.squeeze().cpu().numpy(), (1, 2, 0))
                P_T_pred_numpy = np.transpose(P_T_pred.squeeze().cpu().numpy(), (1, 2, 0))
                T_pred_numpy = np.transpose(T_pred.squeeze().cpu().numpy(), (1, 2, 0))
                A_infinity_pred_numpy = np.transpose(A_infinity_pred.squeeze().cpu().numpy(), (1, 2, 0))
                R_pred_numpy = np.transpose(R_pred.squeeze().cpu().numpy(), (1, 2, 0))
                np.save(os.path.join(P_A_pred_dir, name + '.npy'), P_A_pred_numpy)
                np.save(os.path.join(P_T_pred_dir, name + '.npy'), P_T_pred_numpy)
                np.save(os.path.join(T_pred_dir, name + '.npy'), T_pred_numpy)
                np.save(os.path.join(A_infinity_pred_dir, name + '.npy'), A_infinity_pred_numpy)
                np.save(os.path.join(R_pred_dir, name + '.npy'), R_pred_numpy)
                # use this to output T_hat for visualization
                # T_hat_pred_numpy = np.transpose(T_hat_pred.squeeze().cpu().numpy(), (1, 2, 0))
                # R_hat_pred_numpy = np.transpose(R_hat_pred.squeeze().cpu().numpy(), (1, 2, 0))
                # np.save(os.path.join(T_hat_pred_dir, name + '.npy'), T_hat_pred_numpy)
                # np.save(os.path.join(R_hat_pred_dir, name + '.npy'), R_hat_pred_numpy)


if __name__ == '__main__':
    MODULE = 'full'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--extra_dir', default='', type=str, help='extra dir')
    parser.add_argument('--data_loader_type', default='InferDataLoader', type=str, help='which data loader to use')
    parser.add_argument('--R_only', default=1, type=int, help='only save R without P_A, P_T, T, and A_infinity')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser = subparsers.add_parser("default")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')
    data_loader_class = getattr(module_data, args.data_loader_type)
    data_loader = data_loader_class(data_dir=args.data_dir, extra_dir=args.data_dir)

    # build model architecture
    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # only save R without P_A, P_T, T, and A_infinity
    R_only = args.R_only

    # run the selected func
    if args.func == 'default':
        infer_default()
    else:
        # run the default
        infer_default()
