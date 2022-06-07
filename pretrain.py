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


class Pretrain:
    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.dataloader = DataLoader(
            file_path=args.data_path, max_sequence_length=args.seq_len, min_sequence=5, samples_per_task=args.num_samples, pretraining=True, pretraining_batch_size=128)
        self.pretraining_loader = self.dataloader.pretraining_loader
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        # bert4rec model
        self.model = BERTModel(self.args).to(self.device)

        self._log_dir = args.log_dir
        self._save_dir = os.path.join(args.log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        # whether to use multi step loss
        self._lr = 0.01
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self._lr)
        self.loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

        self.lr_scheduler = optim.lr_scheduler.\
            MultiStepLR(self.optimizer, milestones=[
                        1, 4, 7], gamma=0.1)

        self._train_step = 0

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
        for _ in range(epochs):
            for input, target_rating in tqdm(self.pretraining_loader):
                user_id, product_history, target_product_id,  product_history_ratings = input

                x_inputs = (user_id.to(self.device), product_history.to(
                    self.device),
                    target_product_id.to(
                        self.device),  product_history_ratings.to(self.device))
                target_rating = target_rating.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(
                    x_inputs)
                loss = self.loss_fn(outputs, target_rating)
                loss.backward()
                self.optimizer.step()
                mae_loss = self.mae_loss_fn(outputs, target_rating).item()
                mse_loss = loss.detach().item()

            if self._train_step % LOG_INTERVAL == 0:
                print(
                    f'Epoch {self._train_step}: '
                    f'MAE loss: {mae_loss:.3f} | '
                    f'MSE loss: {mse_loss:.3f} | '
                )
                writer.add_scalar(
                    "train/MSEloss", mse_loss, self._train_step)
                writer.add_scalar(
                    "train/MAEloss", mae_loss, self._train_step)

            if self._train_step % SAVE_INTERVAL == 0:
                self._save_model()
            self._train_step += 1
            self.lr_scheduler.step()
        writer.close()

    def load(self, checkpoint_step):
        target_path = os.path.join(self._save_dir, f"{checkpoint_step}")
        print("Loading checkpoint from", target_path)
        try:
            self.model.bert.load_state_dict(torch.load(target_path))

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        # Save a model to 'save_dir'
        torch.save(self.model.bert.state_dict(),
                   os.path.join(self._save_dir, f"{self._train_step}"))


def main(args):
    pretrain_module = Pretrain(args)

    if args.checkpoint_step > -1:
        pretrain_module.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    pretrain_module.train(epochs=500)


if __name__ == '__main__':
    main(args)
