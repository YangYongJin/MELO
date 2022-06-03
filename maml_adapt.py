from models.bert import BERTModel
from dataloader_adapt import DataLoader
from options import args

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


SAVE_INTERVAL = 50
LOG_INTERVAL = 1
VAL_INTERVAL = 10
NUM_TRAIN_TASKS = 20
NUM_TEST_TASKS = 100
NUM_ITERATIONS = 200


class MAML:
    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.dataloader = DataLoader(
            file_path=args.data_path, max_sequence_length=args.seq_len, min_sequence=5, samples_per_task=args.num_samples)
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        # bert4rec model
        self.model = BERTModel(self.args)

        self.model = self.model.to(self.device)

        self._log_dir = args.log_dir
        self._save_dir = os.path.join(args.log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        self._num_inner_steps = args.num_inner_steps
        self._inner_lr = args.inner_lr
        self._outer_lr = args.outer_lr

        self._train_step = 0

        self.loss_network = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.loss_lr = 0.01
        self.loss_optimizer = optim.Adam(
            self.loss_network.parameters(), lr=self.loss_lr)

        print("Finished initialization")

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self._num_inner_steps)) * (
            1.0 / self._num_inner_steps)
        decay_rate = 1.0 / self._num_inner_steps / \
            self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self._num_inner_steps
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(
                loss_weights[i] - (self._train_step * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self._train_step *
                                (self._num_inner_steps - 1) * decay_rate),
            1.0 - ((self._num_inner_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def _inner_loop(self, theta, support_data, task_info):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            theta (List[Tensor]): current model parameters
            support_data (Tensor): task support set inputs

        Returns:
            phi: optimized weight(state_dict)
        """

        # loss function
        loss_fn = nn.MSELoss()

        # inner loop param
        phi_model = copy.deepcopy(theta)
        optimizer = optim.SGD(phi_model.parameters(), lr=self._inner_lr)

        user_id, product_history, target_product_id,  product_history_ratings, target_rating = support_data
        inputs = user_id.to(self.device), product_history.to(
            self.device), \
            target_product_id.to(
                self.device),  product_history_ratings.to(self.device)
        task_info = task_info.to(self.device)
        task_info_adapt = (task_info-task_info.mean()) / \
            (task_info.std() + 1e-12)
        target_rating = target_rating.to(self.device)
        # inner loop optimization
        for _ in range(self._num_inner_steps + 1):
            optimizer.zero_grad()
            outputs = phi_model(inputs)
            loss = loss_fn(outputs, target_rating)
            loss += self.loss_network(task_info_adapt)[0]
            loss.backward()
            optimizer.step()
        phi = phi_model.state_dict()
        return phi

    def _outer_loop(self, task_batch, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss: mean query MSE loss over the batch
            mae_loss: mean query MAE loss over the batch
        """

        theta = copy.deepcopy(self.model).to(self.device)

        outer_loss_batch = []
        mae_loss_batch = []

        loss_fn = nn.MSELoss()
        mae_loss_fn = nn.L1Loss()
        optimizer = optim.SGD(theta.parameters(), lr=self._outer_lr)
        optimizer.zero_grad()
        self.loss_optimizer.zero_grad()
        for idx, task in enumerate(tqdm(task_batch)):
            support, query, task_info = task
            phi = self._inner_loop(theta, support, task_info)  # do inner loop
            self.model.load_state_dict(phi)
            user_id, product_history, target_product_id,  product_history_ratings, target_rating = query
            inputs = user_id.to(self.device), product_history.to(
                self.device), \
                target_product_id.to(
                    self.device),  product_history_ratings.to(self.device)
            target_rating = target_rating.to(self.device)
            if train:
                self.model.train()
            else:
                self.model.eval()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, target_rating)
            self.model.zero_grad()
            # optimizer.zero_grad()
            loss.backward()

            for k, v in zip(theta.parameters(), self.model.parameters()):
                if idx == 0:
                    k.grad = (v.grad)
                else:
                    k.grad += (v.grad)
            # if train:
            #     optimizer.step()

            mae_loss = mae_loss_fn(outputs, target_rating)
            mae_loss_batch.append(mae_loss.detach().to("cpu").item())
            outer_loss_batch.append(loss.detach().to("cpu").item())

        # Update model with new theta
        if train:
            optimizer.step()
            self.loss_optimizer.step()
            self.model.load_state_dict(theta.state_dict())

        outer_loss = np.mean(outer_loss_batch)
        mae_loss = np.mean(mae_loss_batch)

        return outer_loss, mae_loss

    def train(self, train_steps):
        """Train the MAML.

        Optimizes MAML meta-parameters
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting MAML training at iteration {self._train_step}")
        writer = SummaryWriter()
        val_batches = self.dataloader.generate_task(
            mode="valid", batch_size=50)
        for i in range(1, train_steps+1):
            self._train_step += 1
            train_task = self.dataloader.generate_task(
                mode="train", batch_size=self.batch_size)

            outer_loss, mae_loss = self._outer_loop(
                train_task, train=True)

            if self._train_step % SAVE_INTERVAL == 0:
                self._save_model()

            if i % LOG_INTERVAL == 0:
                print(
                    f'Iteration {self._train_step}: '
                    f'MSE loss: {outer_loss:.3f} | '
                    f'MAE loss: {mae_loss:.3f} | '
                )
                writer.add_scalar(
                    "train/MSEloss", outer_loss, self._train_step)
                writer.add_scalar("train/MAEloss", mae_loss, self._train_step)

            if i % VAL_INTERVAL == 0:
                outer_loss, mae_loss = self._outer_loop(
                    val_batches, train=False)

                print(
                    f'\t-Validation: '
                    f'Val MSE loss: {outer_loss:.3f} | '
                    f'Val MAE loss: {mae_loss:.3f} | '
                )

                writer.add_scalar(
                    "train/MSEloss", outer_loss, self._train_step)
                writer.add_scalar("train/MAEloss", mae_loss, self._train_step)
        writer.close()

    def test(self):
        accuracies = []
        test = [self.val_data.generate_task(
            NUM_TEST_TASKS//10) for _ in range(10)]
        for test_data in test:
            if self.is_plus:
                _, _, accuracy_query = self._outer_loop_plus(
                    test_data, train=False)
            else:
                _, _, accuracy_query = self._outer_loop(test_data, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(10)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        target_path = os.path.join(self._save_dir, f"{checkpoint_step}")
        print("Loading checkpoint from", target_path)
        try:
            self.model.load_state_dict(torch.load(target_path))

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        # Save a model to 'save_dir'
        torch.save(self.model.state_dict(),
                   os.path.join(self._save_dir, f"{self._train_step}"))


def main(args):
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.abspath('.'), "p1_log/")

    print(f'log_dir: {args.log_dir}')

    maml = MAML(
        args
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        maml.train(args.num_train_iterations)

    else:
        maml.test()


if __name__ == '__main__':
    main(args)
