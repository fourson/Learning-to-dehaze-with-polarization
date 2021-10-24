import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self._load_pretrained_weights(**extra_args)  # for loading the pretrained weights of T_model and R_model

    def _load_pretrained_weights(self, **extra_args):
        T_model_checkpoint_path = extra_args.get('T_model_checkpoint_path')
        R_model_checkpoint_path = extra_args.get('R_model_checkpoint_path')
        if T_model_checkpoint_path:
            T_model_checkpoint = torch.load(T_model_checkpoint_path)
            if self.data_parallel:
                self.model.module.T_model.load_state_dict(T_model_checkpoint['model'])
            else:
                self.model.T_model.load_state_dict(T_model_checkpoint['model'])
            print('load T_model_checkpoint ...')
        if R_model_checkpoint_path:
            R_model_checkpoint = torch.load(R_model_checkpoint_path)
            if self.data_parallel:
                self.model.module.R_model.load_state_dict(R_model_checkpoint['model'])
            else:
                self.model.R_model.load_state_dict(R_model_checkpoint['model'])
            print('load R_model_checkpoint ...')

    def _eval_metrics(self, pred, gt):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(pred, gt)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            # (N, 3*C, H, W) GPU tensor
            I_alpha = sample['I_alpha'].to(self.device)
            # (N, C, H, W) GPU tensor
            I = sample['I'].to(self.device)
            delta_I = sample['delta_I'].to(self.device)

            # (N, C, H, W) GPU tensor
            P_A = sample['P_A'].to(self.device)
            P_T = sample['P_T'].to(self.device)
            T = sample['T'].to(self.device)
            A_infinity = sample['A_infinity'].to(self.device)
            R = sample['R'].to(self.device)

            # get network output
            # (N, C, H, W) GPU tensor
            P_A_pred, P_T_pred, T_pred, A_infinity_pred, R_pred = self.model(I_alpha, I, delta_I)

            # visualization
            with torch.no_grad():
                if batch_idx % 100 == 0:
                    # save images to tensorboardX
                    self.writer.add_image('T_pred', make_grid(T_pred))
                    self.writer.add_image('T', make_grid(T))
                    self.writer.add_image('R_pred', make_grid(R_pred))
                    self.writer.add_image('R', make_grid(R))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(P_A_pred, P_A, P_T_pred, P_T, T_pred, T, A_infinity_pred, A_infinity, R_pred, R)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            total_metrics += self._eval_metrics(R_pred, R)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                # (N, 3*C, H, W) GPU tensor
                I_alpha = sample['I_alpha'].to(self.device)
                # (N, C, H, W) GPU tensor
                I = sample['I'].to(self.device)
                delta_I = sample['delta_I'].to(self.device)

                # (N, C, H, W) GPU tensor
                P_A = sample['P_A'].to(self.device)
                P_T = sample['P_T'].to(self.device)
                T = sample['T'].to(self.device)
                A_infinity = sample['A_infinity'].to(self.device)
                R = sample['R'].to(self.device)

                # get network output
                # (N, C, H, W) GPU tensor
                P_A_pred, P_T_pred, T_pred, A_infinity_pred, R_pred = self.model(I_alpha, I, delta_I)

                loss = self.loss(P_A_pred, P_A, P_T_pred, P_T, T_pred, T, A_infinity_pred, A_infinity, R_pred, R)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(T_pred, T)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
