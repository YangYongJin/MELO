from models.bert import BERTModel
from dataloader import DataLoader
from options import args

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# SAVE_INTERVAL = 50
LOG_INTERVAL = 1
VAL_INTERVAL = 25


class MAML:
    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size  # task batch size

        # load dataloader
        self.dataloader = DataLoader(args, pretraining=False)

        # set # of users and # of items
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        # set device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        # define model (theta)
        self.model = BERTModel(self.args).to(self.device)

        # set log and save directories
        self._log_dir = args.log_dir
        self._save_dir = os.path.join(args.log_dir, 'state')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        # whether to use multi step loss
        self.use_multi_step = args.use_multi_step

        # meta hyperparameters
        self._num_inner_steps = args.num_inner_steps
        self._inner_lr = args.inner_lr
        # for bert
        self._outer_lr = args.outer_lr
        # for last fc layers
        self._fc_lr = args.fc_lr

        # user normalized ratings (0, 1)
        self.normalize_loss = self.args.normalize_loss

        # freeze bert model and only update last layers
        if args.freeze_bert:
            self.meta_optimizer = optim.Adam(
                self.model.parameters(), lr=self._fc_lr, weight_decay=args.fc_weight_decay)
        else:
            # apply different learning rate for bert and fc
            self.meta_optimizer = optim.Adam([
                {'params': self.model.bert.parameters()},
                {'params': self.model.dim_reduct.parameters(), 'lr': self._fc_lr,
                 'weight_decay': args.fc_weight_decay},
                {'params': self.model.out.parameters(), 'lr': self._fc_lr,
                 'weight_decay': args.fc_weight_decay}
            ], lr=self._outer_lr)

        # meta learning rate scheduler
        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=args.num_train_iterations, eta_min=1e-6)
        # current epoch
        self._train_step = 0

        # options - using adaptive loss and using weight of adaptive loss
        self.use_adaptive_loss = args.use_adaptive_loss
        self.use_adaptive_loss_weight = (
            args.use_adaptive_loss_weight and self.use_adaptive_loss)
        self.use_lstm = args.use_lstm
        num_loss_layers = None
        # settings for adaptive loss
        if self.use_adaptive_loss:
            self._loss_lr = args.loss_lr
            self.task_info_loss = args.task_info_loss
            num_loss_layers = self.task_info_loss * 1 + 1*args.task_info_rating_mean + 1 * \
                args.task_info_rating_std + 1*args.task_info_num_samples + \
                5*args.task_info_rating_distribution
            self.loss_network = nn.Sequential(
                nn.Linear(num_loss_layers, num_loss_layers),
                nn.ReLU(),
                nn.Linear(num_loss_layers, 1),
            ).to(self.device)
            self.loss_optimizer = optim.Adam(
                self.loss_network.parameters(), lr=self._loss_lr)
            self.loss_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.loss_optimizer, milestones=[
                            500, 800, 950], gamma=0.7)

         # settings for adaptive loss weight
        if self.use_adaptive_loss_weight:
            num_loss_weight_layers = num_loss_layers - self.task_info_loss * 1
            self._task_info_lr = args.task_info_lr
            self.task_info_network = nn.Sequential(
                nn.Linear(num_loss_weight_layers, num_loss_weight_layers),
                nn.ReLU(),
                nn.Linear(num_loss_weight_layers, 1),
            ).to(self.device)
            self.task_info_optimizer = optim.Adam(
                self.task_info_network.parameters(), lr=self._task_info_lr)
            self.task_info_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.task_info_optimizer, milestones=[
                            500, 800, 950], gamma=0.7)

        # use lstm as task information
        if self.use_lstm:
            self._lstm_lr = args.lstm_lr
            self.task_lstm_network = nn.LSTM(
                batch_first=True, input_size=1, hidden_size=8, num_layers=1, dropout=0.3).to(self.device)
            self.task_lstm_optimizer = optim.Adam(
                self.task_lstm_network.parameters(), lr=self._lstm_lr)

            # best results
        self.best_step = 0
        self.best_valid_rmse_loss = 987654321

        print("Finished initialization")

    # per step loss weight for multi step loss function
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

    # update meta paramters
    def update_meta_params(self, query_inputs, query_target_rating, optimizer, mae_loss_fn, phi_model, imp_weight=1):
        '''
        Update gradient values of meta learning parameters
        Args:
            query_inputs: query inputs
            query_target_rating : query target rating
            optimizer: inner loop optimizer
            loss_fn : inner loop loss(mse)
            mae_loss_fn : mae loss
            phi_model : current inner loop paramters phi
            imp_weight : importance weight vector for gradients(multi step)
        '''
        # zero grad
        loss_fn = nn.MSELoss()
        optimizer.zero_grad()

        # forward propagate on query data
        outputs = phi_model(query_inputs)

        # compute loss
        if self.normalize_loss:
            query_loss = loss_fn(outputs, query_target_rating/5.0)
            mae_loss = mae_loss_fn(
                outputs.clone().detach()*5, query_target_rating)
            query_output_loss = loss_fn(
                outputs.clone().detach()*5, query_target_rating)
        else:
            query_loss = loss_fn(outputs, query_target_rating)
            mae_loss = mae_loss_fn(
                outputs.clone().detach(), query_target_rating)
            query_output_loss = query_loss.clone().detach()

        query_loss.backward()

        # update gradients of meta paramters
        for k, v in zip(self.model.parameters(), phi_model.parameters()):
            if k.requires_grad == True:
                if k.grad == None:
                    k.grad = imp_weight*(v.grad)
                else:
                    k.grad += imp_weight*(v.grad)

        return query_output_loss, mae_loss

    # inner loop optimization
    def _inner_loop(self, support_data, task_info, query_inputs, query_target_rating, imp_vecs, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            support_data: support data
            task_info: task information
            query_inputs: query data
            query_target_rating: query target
            imp_vecs: important vectors for multi step loss
            train: train params

        Returns:
            query_loss: query mse loss
            mae_loss: query mae loss
        """

        # loss functions
        loss_fn = nn.MSELoss(reduction='none')
        mae_loss_fn = nn.L1Loss()

        # inner loop parameters phi
        phi_model = copy.deepcopy(self.model)

        # if freeze bert, do not update bert
        if self.args.freeze_bert:
            for param in phi_model.bert.parameters():
                param.requires_grad = False

        # inner loop optimizers
        optimizer = optim.SGD(phi_model.parameters(), lr=self._inner_lr)

        # GPU enabling
        user_id, product_history, target_product_id,  product_history_ratings, target_rating = support_data
        inputs = user_id.to(self.device), product_history.to(
            self.device), \
            target_product_id.to(
                self.device),  product_history_ratings.to(self.device)
        task_info = task_info.to(self.device)

        target_rating = target_rating.to(self.device)

        # inner loop optimization
        for step in range(self._num_inner_steps):

            # forward propagate on support set
            optimizer.zero_grad()
            outputs = phi_model(inputs)

            # compute loss
            if self.normalize_loss:
                loss = loss_fn(outputs, target_rating/5.0)
            else:
                loss = loss_fn(outputs, target_rating)

            # adaptive loss
            if self.use_adaptive_loss:
                # normalize task information

                if self.use_lstm:
                    task_input = torch.cat(
                        (inputs[3], target_rating), dim=1).reshape(-1, self.args.max_seq_len, 1)/5.0
                    b, t, _ = task_input.shape
                    h0 = torch.randn(1, b, 8).to(self.device)
                    c0 = torch.randn(1, b, 8).to(self.device)
                    task_info, (h_out, c_out) = self.task_lstm_network(
                        task_input, (h0, c0))
                    task_info = task_info[:, -1, :]
                    task_info = torch.cat((loss, task_info), dim=1)
                    # task_info_adapt = (task_info-task_info.mean(dim=1, keepdim=True)) / \
                    #     (task_info.std(dim=1, keepdim=True) + 1e-5)
                    loss = self.loss_network(task_info)
                    loss = torch.mean(loss)

                else:
                    loss = torch.mean(loss)
                    task_info_step = torch.cat(
                        (loss.reshape(1), task_info))
                    task_info_adapt = (task_info_step-task_info_step.mean()) / \
                        (task_info_step.std() + 1e-5)
                    if self.use_adaptive_loss_weight:
                        weight = self.task_info_network(task_info)[0]
                        loss += weight * self.loss_network(task_info_adapt)[0]
                    else:
                        loss += self.loss_network(task_info_adapt)[0]

            # update inner loop paramters phi
            loss.backward()
            optimizer.step()

            ##### multi step loss - update meta paramters ######
            if self.use_multi_step and self._train_step < self.args.multi_step_loss_num_epochs and train:
                query_loss, mae_loss = self.update_meta_params(
                    query_inputs, query_target_rating, optimizer, mae_loss_fn, phi_model, imp_vecs[step])

            ##### Fo-maml loss - update meta paramters at last step ####
            ### also use this step for valid set ###
            else:
                # at last step
                if step == self._num_inner_steps - 1:
                    if train:
                        phi_model.train()
                    else:
                        phi_model.eval()

                    query_loss, mae_loss = self.update_meta_params(
                        query_inputs, query_target_rating, optimizer, mae_loss_fn, phi_model)

        return query_loss, mae_loss

    # outer loop
    def _outer_loop(self, task_batch, train=None):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            mse_loss: mean query MSE loss over the batch
            rmse_loss: mean query RMSE loss over the batch
            mae_loss: mean query MAE loss over the batch
        """
        # get importance weight
        imp_vecs = self.get_per_step_loss_importance_vector()

        mse_loss_batch = []
        mae_loss_batch = []

        self.meta_optimizer.zero_grad()
        if self.use_adaptive_loss:
            self.loss_optimizer.zero_grad()
        if self.use_lstm:
            self.task_lstm_optimizer.zero_grad()
        if self.use_adaptive_loss_weight:
            self.task_info_optimizer.zero_grad()

        # loop through task batch
        for idx, task in enumerate(tqdm(task_batch)):
            # query data gpu loading
            support, query, task_info = task
            user_id, product_history, target_product_id,  product_history_ratings, target_rating = query
            query_inputs = user_id.to(self.device), product_history.to(
                self.device), \
                target_product_id.to(
                    self.device),  product_history_ratings.to(self.device)
            query_target_rating = target_rating.to(self.device)

            # inner loop operation
            loss, mae_loss = self._inner_loop(
                support, task_info, query_inputs, query_target_rating, imp_vecs, train)  # do inner loop

            # collect loss data
            mse_loss_batch.append(loss.detach().to("cpu").item())
            mae_loss_batch.append(mae_loss.detach().to("cpu").item())

        # Update meta parameters
        if train:
            total_norm = 0.0
            for p in self.model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(total_norm)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=100)
            self.meta_optimizer.step()
            # self.meta_lr_scheduler.step()
            if self.use_adaptive_loss:
                total_norm = 0.0
                for p in self.loss_network.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(total_norm)
                torch.nn.utils.clip_grad_norm_(
                    self.loss_network.parameters(), max_norm=100)
                self.loss_optimizer.step()
                # self.loss_lr_scheduler.step()
            if self.use_lstm:
                total_norm = 0.0
                for p in self.task_lstm_network.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(total_norm)
                torch.nn.utils.clip_grad_norm_(
                    self.task_lstm_network.parameters(), max_norm=100)
                self.task_lstm_optimizer.step()
            if self.use_adaptive_loss_weight:
                self.task_info_optimizer.step()
                # self.task_info_lr_scheduler.step()

        # set results
        mse_loss = np.mean(mse_loss_batch)
        rmse_loss = np.sqrt(mse_loss)
        mae_loss = np.mean(mae_loss_batch)

        return mse_loss, rmse_loss, mae_loss

    def train(self, train_steps):
        """Train the MAML.

        Optimizes MAML meta-parameters
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting MAML training at iteration {self._train_step}")

        # define tensorboard writer
        writer = SummaryWriter(log_dir=self._log_dir)

        # set validation tasks
        val_batches = self.dataloader.generate_task(
            mode="valid", batch_size=300, normalized=self.normalize_loss, use_label=self.args.use_label)

        # iteration
        for i in range(1, train_steps+1):
            self._train_step += 1

            # generate train task batch
            train_task = self.dataloader.generate_task(
                mode="train", batch_size=self.batch_size, normalized=self.normalize_loss, use_label=self.args.use_label)

            # update meta paramters and return losses
            mse_loss, rmse_loss, mae_loss = self._outer_loop(
                train_task, train=True)

            # looging
            if i % LOG_INTERVAL == 0:
                print(
                    f'Iteration {self._train_step}: '
                    f'MSE loss: {mse_loss:.4f} | '
                    f'RMSE loss: {rmse_loss:.4f} | '
                    f'MAE loss: {mae_loss:.4f} | '
                )
                writer.add_scalar("train/MSEloss", mse_loss, self._train_step)
                writer.add_scalar("train/RMSEloss",
                                  rmse_loss, self._train_step)
                writer.add_scalar("train/MAEloss", mae_loss, self._train_step)

            # evaluate validation set
            if i % VAL_INTERVAL == 0:
                mse_loss, rmse_loss, mae_loss = self._outer_loop(
                    val_batches, train=False)

                print(
                    f'\tValidation: '
                    f'Val MSE loss: {mse_loss:.4f} | '
                    f'Val RMSE loss: {rmse_loss:.4f} | '
                    f'Val MAE loss: {mae_loss:.4f} | '
                )

                # Save the best model wrt valid rmse loss
                if self.best_valid_rmse_loss > rmse_loss:
                    self.best_valid_rmse_loss = rmse_loss
                    self.best_step = i
                    self._save_model()
                    print(
                        f'........Model saved (step: {self.best_step} | RMSE loss: {rmse_loss:.4f})')

                writer.add_scalar("valid/MSEloss", mse_loss, self._train_step)
                writer.add_scalar("valid/RMSEloss",
                                  rmse_loss, self._train_step)
                writer.add_scalar("valid/MAEloss", mae_loss, self._train_step)
        writer.close()

        print("-------------------------------------------------")
        print("Model with the best validation RMSE loss is saved.")
        print(f'Best step: {self.best_step}')
        print(f'Best RMSE loss: {self.best_valid_rmse_loss:.4f}')
        print("Done.")

    def test(self):
        '''
            Test on test batches
        '''
        test_batches = self.dataloader.generate_task(
            mode="test", batch_size=self.args.num_test_data, normalized=self.normalize_loss, use_label=self.args.use_label)
        mse_loss, rmse_loss, mae_loss = self._outer_loop(
            test_batches, train=False)
        print(
            f'\tTest: '
            f'Test MSE loss: {mse_loss:.4f} | '
            f'Test RMSE loss: {rmse_loss:.4f} | '
            f'Test MAE loss: {mae_loss:.4f} | '
        )

    def test_baseline(self):
        '''
            Test on test batches
        '''

        rating_lst = sum(self.dataloader.train_set['rating'].tolist(), [])
        mean_rating = np.mean(rating_lst, dtype='float32')
        print(mean_rating)
        test_batches = self.dataloader.generate_task(
            mode="test", batch_size=self.args.num_test_data, normalized=self.normalize_loss, use_label=self.args.use_label)
        mse_loss_batch = []
        mae_loss_batch = []
        for idx, task in enumerate(tqdm(test_batches)):
            # query data gpu loading
            _, query, _ = task
            _, _, _,  _, target_rating = query
            query_target_rating = target_rating.to(self.device)
            query_predict_rating = torch.ones_like(
                target_rating).to(self.device) * mean_rating
            mse_loss = nn.MSELoss()(query_predict_rating, query_target_rating).to("cpu").item()
            mae_loss = nn.L1Loss()(query_predict_rating, query_target_rating).to("cpu").item()
            mse_loss_batch.append(mse_loss)
            mae_loss_batch.append(mae_loss)

        mse_loss = np.mean(mse_loss_batch)
        rmse_loss = np.sqrt(mse_loss)
        mae_loss = np.mean(mae_loss_batch)
        print(
            f'\tTest: '
            f'Test MSE loss: {mse_loss:.4f} | '
            f'Test RMSE loss: {rmse_loss:.4f} | '
            f'Test MAE loss: {mae_loss:.4f} | '
        )

    def load(self, checkpoint_step):
        '''
            load meta paramters
        '''
        target_path = os.path.join(
            self._save_dir, f"{checkpoint_step}_best.pt")
        print("Loading checkpoint from", target_path)
        try:
            if torch.cuda.is_available():
                def map_location(storage, loc): return storage.cuda()
            else:
                map_location = 'cpu'
            checkpoint = torch.load(target_path, map_location=map_location)
            self.model.load_state_dict(checkpoint['meta_model'])
            if self.use_adaptive_loss:
                self.loss_network.load_state_dict(checkpoint['loss_model'])
            if self.use_adaptive_loss_weight:
                self.task_info_network.load_state_dict(
                    checkpoint['loss_weight_model'])

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        '''
            save meta paramters
        '''
        save_path = os.path.join(self._save_dir, f"{self._train_step}_best.pt")
        model_dict = {
            'meta_model': self.model.state_dict()
        }
        if self.use_adaptive_loss:
            model_dict['loss_model'] = self.loss_network.state_dict()
        if self.use_adaptive_loss_weight:
            model_dict['loss_weight_model'] = self.task_info_network.state_dict()
        torch.save(model_dict, save_path)

    def load_pretrained_bert(self, filename):
        '''
            load pretrained bert model
        '''
        pretrained_path = os.path.join('./pretrained', filename)
        print("Loading Pretrained Model")
        try:
            if torch.cuda.is_available():
                def map_location(storage, loc): return storage.cuda()
            else:
                map_location = 'cpu'

            if self.args.load_save_bert:
                self.model.bert.load_state_dict(torch.load(
                    pretrained_path, map_location=map_location))
            else:
                self.model.load_state_dict(torch.load(
                    pretrained_path, map_location=map_location))

        except:
            raise ValueError(
                f'No Pretrained Model or something goes wrong.')

    def freeze_bert(self):
        '''
            freeze bert
        '''
        for param in self.model.bert.parameters():
            param.requires_grad = False


def main(args):
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.abspath('.'), "log/")

    print(f'log_dir: {args.log_dir}')

    maml = MAML(
        args
    )

    if args.load_pretrained:
        dir = os.listdir('./pretrained')
        if len(dir) != 0:
            for filename in dir:
                if 'best' in filename:
                    maml.load_pretrained_bert(filename)
                    break
            if args.freeze_bert:
                maml.freeze_bert()
        else:
            print("No pretrained model - skip loading")

    else:
        print('Pretrained Model loading skipped')

    if args.checkpoint_step > -1:
        maml._train_step = args.checkpoint_step
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        maml.train(args.num_train_iterations)
        # maml.test_baseline()

    else:
        maml.test()


if __name__ == '__main__':
    main(args)
