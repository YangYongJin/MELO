from models.bert import BERTModel
from dataloader import DataLoader
from options import args

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
SAVE_INTERVAL = 100
LOG_INTERVAL = 1
VAL_INTERVAL = 25


class Pretrain:
    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.dataloader = DataLoader(args, pretraining=True)
        self.pretraining_train_loader = self.dataloader.pretraining_train_loader
        self.pretraining_valid_loader = self.dataloader.pretraining_valid_loader
        self.pretraining_test_loader = self.dataloader.pretraining_test_loader
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        # bert4rec model
        self.model = BERTModel(self.args).to(self.device)

        self._log_dir = args.pretrain_log_dir
        self._save_dir = os.path.join(args.pretrain_log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        # whether to use multi step loss
        self._lr = args.pretraining_lr
        self.optimizer = optim.Adam([
            {'params': self.model.bert.parameters()},
            {'params': self.model.dim_reduct.parameters(), 'lr': args.fc_lr,
             'weight_decay': args.fc_weight_decay},
            {'params': self.model.out.parameters(), 'lr': args.fc_lr,
             'weight_decay': args.fc_weight_decay}
        ], lr=self._lr)

        self.loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.best_valid_rmse_loss = 987654321
        self.best_step = 0

        self.normalize_loss = self.args.normalize_loss

        self.lr_scheduler = optim.lr_scheduler.\
            MultiStepLR(self.optimizer, milestones=[
                        400, 700, 900], gamma=0.1)

        ##### load and save whole model or only bert #####
        self.load_save_bert = args.load_save_bert

        self._train_step = 0

    def epoch_step(self, data_loader, train=True):
        '''
            do one epoch step
        '''
        mse_losses = []
        mae_losses = []
        rmse_losses = []
        if train:
            self.model.train()
        else:
            self.model.eval()
        # one epoch opeartion
        for input, target_rating in tqdm(data_loader):
            user_id, product_history, target_product_id,  product_history_ratings = input

            B, S, T = product_history.shape
            user_id = user_id.view(-1, 1)
            product_history = product_history.view(-1, T)
            target_product_id = target_product_id.view(-1, 1)
            product_history_ratings = product_history_ratings.view(-1, T)
            target_rating = target_rating.view(-1, 1)

            # gpu loading
            x_inputs = (user_id.to(self.device), product_history.to(
                self.device),
                target_product_id.to(
                    self.device),  product_history_ratings.to(self.device))
            target_rating = target_rating.to(self.device)
            self.optimizer.zero_grad()

            # forward prop
            outputs = self.model(
                x_inputs)

            # compute loss
            if self.normalize_loss:
                loss = self.loss_fn(outputs, target_rating/5.0)
                mse_loss = self.loss_fn(
                    outputs.clone().detach()*5, target_rating)
                mae_loss = self.mae_loss_fn(
                    outputs.clone().detach()*5, target_rating)
                rmse_loss = torch.sqrt(mse_loss)
            else:
                loss = self.loss_fn(outputs, target_rating)
                mse_loss = self.loss_fn(
                    outputs.clone().detach(), target_rating)
                mae_loss = self.mae_loss_fn(
                    outputs.clone().detach(), target_rating)
                rmse_loss = torch.sqrt(mse_loss)

            # update paramters
            if train:
                loss.backward()
                self.optimizer.step()
            mse_losses.append(mse_loss.item())
            mae_losses.append(mae_loss.item())
            rmse_losses.append(rmse_loss.item())

        # set results
        mae_loss = np.mean(mae_losses)
        mse_loss = np.mean(mse_losses)
        rmse_loss = np.mean(rmse_losses)

        return mse_loss, mae_loss, rmse_loss

    def train(self, epochs):
        """Train the MAML.

        Optimizes MAML meta-parameters
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting MAML training at iteration {self._train_step}")
        writer = SummaryWriter(log_dir=self._log_dir)
        for epoch in range(epochs):
            mse_loss, mae_loss, rmse_loss = self.epoch_step(
                self.pretraining_train_loader)

            if self._train_step % LOG_INTERVAL == 0:
                print(
                    f'Epoch {self._train_step}: '
                    f'MSE loss: {mse_loss:.3f} | '
                    f'RMSE loss: {rmse_loss:.3f} | '
                    f'MAE loss: {mae_loss:.3f} | '
                )
                writer.add_scalar(
                    "train/MSEloss", mse_loss, self._train_step)
                writer.add_scalar(
                    "train/MAEloss", mae_loss, self._train_step)

            if epoch % VAL_INTERVAL == 0:
                mse_loss, mae_loss, rmse_loss = self.epoch_step(
                    self.pretraining_valid_loader, train=False)

                print(
                    f'\tValidation: '
                    f'Val MSE loss: {mse_loss:.3f} | '
                    f'Val RMSE loss: {rmse_loss:.3f} | '
                    f'Val MAE loss: {mae_loss:.3f} | '
                )

                # Save the best model wrt valid rmse loss
                if self.best_valid_rmse_loss > rmse_loss:
                    self.best_valid_rmse_loss = rmse_loss
                    self.best_step = epoch
                    self._save_model()
                    print(
                        f'........Model saved (step: {self.best_step} | RMSE loss: {rmse_loss:.3f})')

            self._train_step += 1
            # self.lr_scheduler.step()
        writer.close()

    def load(self, checkpoint_step):
        '''
            load model
        '''
        target_path = os.path.join(self._save_dir, f"{checkpoint_step}_best")
        print("Loading checkpoint from", target_path)
        try:
            # set device location
            if torch.cuda.is_available():
                def map_location(storage, loc): return storage.cuda()
            else:
                map_location = 'cpu'

            if self.load_save_bert:
                self.model.bert.load_state_dict(
                    torch.load(target_path, map_location=map_location)
                )
            else:
                self.model.load_state_dict(
                    torch.load(target_path, map_location=map_location)
                )

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        '''
            save model
        '''
        # Save a model to 'save_dir'
        if self.load_save_bert:
            torch.save(self.model.bert.state_dict(),
                       os.path.join(self._save_dir, f"{self._train_step}_best"))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self._save_dir, f"{self._train_step}_best"))

    def test(self):
        '''
            test
        '''
        mse_loss, mae_loss, rmse_loss = self.epoch_step(
            self.pretraining_test_loader, train=False)

        print(
            f'\tTest: '
            f'Test MSE loss: {mse_loss:.3f} | '
            f'Test RMSE loss: {rmse_loss:.3f} | '
            f'Test MAE loss: {mae_loss:.3f} | '
        )


def main(args):
    pretrain_module = Pretrain(args)

    if args.checkpoint_step > -1:
        pretrain_module._train_step = args.checkpoint_step
        pretrain_module.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if args.test:
        pretrain_module.test()
    else:
        pretrain_module.train(epochs=args.pretrain_epochs)


if __name__ == '__main__':
    main(args)
