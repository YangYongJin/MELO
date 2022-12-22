from models import model_factory
from dataloader import DataLoader
from options import args
import wandb

from models.meta_loss_model import MetaTaskLstmNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
SAVE_INTERVAL = 100
LOG_INTERVAL = 1
VAL_INTERVAL = 1


class Basic:
    def __init__(self, args):

        # load dataloaders
        self.args = args
        self.batch_size = args.batch_size
        self.dataloader = DataLoader(args, pretraining=True)
        self.pretraining_train_loader = self.dataloader.pretraining_train_loader
        self.pretraining_valid_loader = self.dataloader.pretraining_valid_loader
        self.pretraining_test_loader = self.dataloader.pretraining_test_loader
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        # device settings
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.args.device = self.device
        # get basic model
        self.model = model_factory(self.args).to(self.device)

        # set logging and saving folders
        self._log_dir = args.pretrain_log_dir
        self._save_dir = os.path.join(args.pretrain_log_dir, 'state')
        self._embedding_dir = os.path.join(args.pretrain_log_dir, 'embedding')
        self._pretrained_dir = os.path.join(
            args.pretrain_log_dir, 'pretrained')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._embedding_dir, exist_ok=True)
        os.makedirs(self._pretrained_dir, exist_ok=True)

        # hyperparamters and optimizers
        self._lr = args.pretraining_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        self.loss_fn = nn.MSELoss(reduction='none')
        self.mae_loss_fn = nn.L1Loss()

        # for learned weighted loss
        self.use_lstm = args.use_learned_loss_baseline

        # use lstm as weighted loss
        if self.use_lstm:
            self._lstm_lr = args.lstm_lr
            lstm_hidden = args.lstm_hidden
            self.task_lstm_network = MetaTaskLstmNetwork(
                input_size=args.lstm_input_size, lstm_hidden=lstm_hidden, num_lstm_layers=args.lstm_num_layers, lstm_out=0, device=self.device, use_softmax=args.use_softmax).to(self.device)
            self.task_lstm_optimizer = optim.Adam(
                self.task_lstm_network.parameters(), lr=self._lstm_lr)
            self.lstm_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.task_lstm_optimizer, T_max=args.pretrain_epochs, eta_min=1e-2)

        self.best_valid_rmse_loss = 987654321
        self.best_step = 0

        # normalize ratings to get range from 0 to 1
        self.normalize_loss = self.args.normalize_loss

        self._train_step = 0

    def epoch_step(self, data_loader, train=True):
        '''
            do one epoch step
            Args:
                dataloader : data to train or evaluate
                train: whether to train or evaluate
            return:
                mse_loss: mean query MSE loss over the batch
                rmse_loss: mean query RMSE loss over the batch
                mae_loss: mean query MAE loss over the batch
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
            if self.use_lstm:
                self.task_lstm_optimizer.zero_grad()

            # forward prop
            outputs = self.model(
                x_inputs)
            gt = torch.cat(
                (x_inputs[3], target_rating), dim=1)
            mask = (gt != 0)

            # compute loss
            if self.normalize_loss:
                loss = self.loss_fn(outputs*mask, gt*mask/5.0)
                if self.use_lstm:
                    task_input = torch.cat(
                        (x_inputs[3], target_rating), dim=1)
                    task_info = self.task_lstm_network(
                        task_input).squeeze()
                    adapt_loss = loss * task_info * mask
                    loss = adapt_loss.sum()/torch.count_nonzero(adapt_loss)
                else:
                    loss = loss.sum()/torch.count_nonzero(loss)
                mse_loss = torch.mean(self.loss_fn(
                    outputs[:, -1:].clone().detach()*5, target_rating))
                mae_loss = self.mae_loss_fn(
                    outputs[:, -1:].clone().detach()*5, target_rating)
                rmse_loss = torch.sqrt(mse_loss)

            else:
                loss = self.loss_fn(outputs*mask, gt*mask)
                if self.use_lstm:
                    task_input = torch.cat(
                        (x_inputs[3], target_rating), dim=1)
                    task_info = self.task_lstm_network(
                        task_input).squeeze()
                    adapt_loss = loss * task_info * mask
                    loss = adapt_loss.sum()/torch.count_nonzero(adapt_loss)
                else:
                    loss = loss.sum()/torch.count_nonzero(loss)
                mse_loss = torch.mean(self.loss_fn(
                    outputs[:, -1:].clone().detach(), target_rating))
                mae_loss = self.mae_loss_fn(
                    outputs[:, -1:].clone().detach(), target_rating)
                rmse_loss = torch.sqrt(mse_loss)

            # update paramters
            if train:
                loss.backward()
                self.optimizer.step()
                if self.use_lstm:
                    self.task_lstm_optimizer.step()
                    self.lstm_lr_scheduler.step()
            mse_losses.append(mse_loss.item())
            mae_losses.append(mae_loss.item())
            rmse_losses.append(rmse_loss.item())

        # set results
        mae_loss = np.mean(mae_losses)
        mse_loss = np.mean(mse_losses)
        rmse_loss = np.mean(rmse_losses)

        return mse_loss, mae_loss, rmse_loss

    def train(self, epochs):
        """Train the Basic.

        Optimizes Basic models
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting Basic model training at iteration {self._train_step}")

        # initialize wandb project
        wandb.init(project=f"BASE-TRAIN-{self.args.model}-{self.args.mode}")

        # define tensorboard writer and wandb config
        # writer = SummaryWriter(log_dir=self._log_dir)
        wandb.config.update(self.args)

        for epoch in range(epochs):
            mse_loss, mae_loss, rmse_loss = self.epoch_step(
                self.pretraining_train_loader)

            if self._train_step % LOG_INTERVAL == 0:
                print(
                    f'Epoch {self._train_step}: '
                    f'MSE loss: {mse_loss:.4f} | '
                    f'RMSE loss: {rmse_loss:.4f} | '
                    f'MAE loss: {mae_loss:.4f} | '
                )
                # writer.add_scalar(
                #     "train/MSEloss", mse_loss, self._train_step)
                # writer.add_scalar(
                #     "train/MAEloss", mae_loss, self._train_step)

            if epoch % VAL_INTERVAL == 0:
                mse_loss, mae_loss, rmse_loss = self.epoch_step(
                    self.pretraining_valid_loader, train=False)

                print(
                    f'\tValidation: '
                    f'Val MSE loss: {mse_loss:.4f} | '
                    f'Val RMSE loss: {rmse_loss:.4f} | '
                    f'Val MAE loss: {mae_loss:.4f} | '
                )
                wandb.log({"loss": rmse_loss})

                # Save the best model wrt valid rmse loss
                if self.best_valid_rmse_loss > rmse_loss:
                    self.best_valid_rmse_loss = rmse_loss
                    self.best_step = epoch
                    self._save_model()
                    print(
                        f'........Model saved (step: {self.best_step} | RMSE loss: {rmse_loss:.4f})')

            self._train_step += 1
        # writer.close()
        print("-------------------------------------------------")
        print("Model with the best validation RMSE loss is saved.")
        print(f'Best step: {self.best_step}')
        print(f'Best RMSE loss: {self.best_valid_rmse_loss:.4f}')
        print("Done.")

    def load(self, checkpoint_step):
        '''
            load model from checkpoint_step
        '''
        target_path = os.path.join(
            self._save_dir, f"{self.args.model}_{checkpoint_step}_no_meta_best")
        print("Loading checkpoint from", target_path)
        try:
            # set device location
            if torch.cuda.is_available():
                def map_location(storage, loc): return storage.cuda()
            else:
                map_location = 'cpu'

            self.model.load_state_dict(
                torch.load(target_path, map_location=map_location)
            )

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        '''
            save models
        '''
        if self.args.save_pretrained:
            if self.args.model == 'sasrec' or self.args.model == 'bert4rec':
                torch.save(self.model.bert.bert_embedding.state_dict(),
                           os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}_{self.args.bert_hidden_units}_{self.args.bert_num_blocks}_{self.args.bert_num_heads}"))
            else:
                torch.save(self.model.embedding.state_dict(),
                           os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}"))

            # Save a model to 'pretrained_dir'
            torch.save(self.model.state_dict(),
                       os.path.join(self._pretrained_dir, f"{self.args.model}_pretrained_{self.args.mode}_{self.args.bert_hidden_units}_{self.args.bert_num_blocks}_{self.args.bert_num_heads}"))
        else:
            # Save a model to 'save_dir'
            torch.save(self.model.state_dict(),
                       os.path.join(self._save_dir, f"{self.args.model}_{self._train_step}_no_meta_best"))

    def test(self):
        '''
            test on basic models
        '''

        # initialize wandb project
        wandb.init(project=f"BASE-TEST-{self.args.model}-{self.args.mode}")

        # define wandb config
        wandb.config.update(self.args)

        mse_loss, mae_loss, rmse_loss = self.epoch_step(
            self.pretraining_test_loader, train=False)

        print(
            f'\tTest: '
            f'Test RMSE loss: {rmse_loss:.4f} | '
            f'Test MAE loss: {mae_loss:.4f} | '
        )
        wandb.log({
            "Test RMSE loss": rmse_loss,
            "Test MAE loss": mae_loss
        })


def main(args):
    basic_model = Basic(args)

    if args.checkpoint_step > -1:
        basic_model._train_step = args.checkpoint_step
        basic_model.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if args.test:
        basic_model.test()
    else:
        basic_model.train(epochs=args.pretrain_epochs)


if __name__ == '__main__':
    main(args)
