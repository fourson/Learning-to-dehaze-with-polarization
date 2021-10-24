import os
import sys
import json
import argparse
import importlib
from functools import partial

import torch


def train(config, resume, **extra_args):
    # prepare logger
    train_logger = Logger()

    # setup data_loader instances
    data_loader_class = getattr(module_data, config['data_loader']['type'])
    data_loader = data_loader_class(**config['data_loader']['args'])
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # show model structure
    print(model)

    # get function handles of loss and metrics
    loss = partial(getattr(module_loss, config['loss']['type']), **config['loss']['args'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer for model parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer = optimizer_class(trainable_params, **config['optimizer']['args'])

    # build learning rate scheduler for optimizer
    lr_scheduler_class = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])
    if config['lr_scheduler']['type'] == 'MultiplicativeLR':
        lr_lambda = util.get_lr_lambda(config['lr_scheduler']['args']['lr_lambda_tag'])
        lr_scheduler = lr_scheduler_class(optimizer, lr_lambda)
    else:
        lr_scheduler = lr_scheduler_class(optimizer, **config['lr_scheduler']['args'])

    # build trainer and train the network
    trainer_class = getattr(module_trainer, config['trainer']['type'])
    trainer = trainer_class(config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                            valid_data_loader, train_logger, **extra_args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PolDehaze')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args, extra_args_raw = parser.parse_known_args()
    extra_args_key = [key.strip('-') for key in extra_args_raw[0::2]]
    extra_args_value = extra_args_raw[1::2]
    extra_args = dict(zip(extra_args_key, extra_args_value))

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH

    config = args.config
    resume = args.resume
    device = args.device

    if config:
        # load config file
        with open(config) as handle:
            config = json.load(handle)
    elif resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    from utils.logger import Logger
    from utils import util

    module = config['module']
    postfix = '_' + module
    module_data = importlib.import_module('.data_loader' + postfix, package='data_loader')
    module_arch = importlib.import_module('.model' + postfix, package='model')
    module_loss = importlib.import_module('.loss' + postfix, package='model')
    module_metric = importlib.import_module('.metric' + postfix, package='model')
    module_trainer = importlib.import_module('.trainer' + postfix, package='trainer')

    train(config, resume, **extra_args)
