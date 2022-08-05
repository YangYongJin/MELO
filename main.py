from multiprocessing import reduction
from models import model_factory
from models.meta_loss_model import MetaLossNetwork, MetaTaskMLPNetwork, MetaTaskLstmNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from dataloader import DataLoader
from options import args
import math
import wandb

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
VAL_INTERVAL = 50


class MAML:
    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size  # task batch size
        self.val_size = args.val_size  # task batch size

        self.val_log_interval = args.log_interval

        # load dataloader
        self.dataloader = DataLoader(args, pretraining=False)

        # set # of users and # of items
        self.args.num_users = self.dataloader.num_users
        self.args.num_items = self.dataloader.num_items

        # set device
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.args.device = self.device

        # define model (theta)
        self.model = model_factory(self.args).to(self.device)

        # model_parameters = filter(
        #     lambda p: p.requires_grad, self.model.bert.bert_embedding.parameters())

        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('# of params', params)
        # 1/0

        # set log and save directories
        self._log_dir = args.log_dir
        self._save_dir = os.path.join(args.log_dir, 'state')
        self._embedding_dir = os.path.join(args.pretrain_log_dir, 'embedding')
        self._pretrained_dir = os.path.join(
            args.pretrain_log_dir, 'pretrained')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._save_dir, exist_ok=True)

        if args.load_pretrained_embedding:
            self._load_pretrained_embedding()
        elif args.load_pretrained:
            self._load_pretrained()

        # whether to use multi step loss
        self.use_multi_step = args.use_multi_step

        # meta hyperparameters
        self._num_inner_steps = args.num_inner_steps
        self._inner_lr = args.inner_lr
        # for meta model
        self._outer_lr = args.outer_lr

        # user normalized ratings (0, 1)
        self.normalize_loss = args.normalize_loss
        self.use_shrinkage_loss = args.use_shrinkage_loss

        # inner loop optimizer
        # self.inner_loop_optimizer = GradientDescentLearningRule(
        #     self.device, learning_rate=self._inner_lr)
        self._use_learnable_params = args.use_learnable_params
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=self.device, total_num_inner_loop_steps=self._num_inner_steps, use_learnable_learning_rates=self._use_learnable_params, init_learning_rate=self._inner_lr)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.model.named_parameters()))
        if self._use_learnable_params:
            self._learning_lr = args.learn_lr
            self.lr_optimizer = optim.Adam(
                self.inner_loop_optimizer.parameters(), lr=self._learning_lr)

        # meta optimizer
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), lr=self._outer_lr)

        # meta learning rate scheduler
        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=args.num_train_iterations, eta_min=args.min_outer_lr)
        # current epoch
        self._train_step = 0

        # options - using adaptive loss and using weight of adaptive loss
        self.use_adaptive_loss = args.use_adaptive_loss
        self.use_adaptive_loss_weight = (
            args.use_mlp and self.use_adaptive_loss)
        self.use_lstm = (
            args.use_lstm and self.use_adaptive_loss)
        num_loss_dims = None
        # settings for adaptive loss
        if self.use_adaptive_loss:
            self._loss_lr = args.loss_lr
            num_loss_dims = args.max_seq_len
            self.loss_network = MetaLossNetwork(
                self._num_inner_steps, num_loss_dims, args.loss_num_layers).to(self.device)
            self.loss_optimizer = optim.Adam(
                self.loss_network.parameters(), lr=self._loss_lr, weight_decay=args.loss_weight_decay)
            self.loss_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.loss_optimizer, milestones=[
                            500, 800, 950], gamma=0.7)

         # settings for adaptive loss weight
        if self.use_adaptive_loss_weight:
            num_loss_weight_dims = 5
            self._task_info_lr = args.task_info_lr
            self.task_info_network = MetaTaskMLPNetwork(
                num_loss_weight_dims, use_softmax=args.use_softmax).to(self.device)
            self.task_info_optimizer = optim.Adam(
                self.task_info_network.parameters(), lr=self._task_info_lr)
            self.task_info_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.task_info_optimizer, milestones=[
                            500, 800, 950], gamma=0.7)

        # use lstm as task information
        if self.use_lstm:
            self._lstm_lr = args.lstm_lr
            lstm_hidden = args.lstm_hidden
            if lstm_hidden < num_loss_dims:
                lstm_hidden = num_loss_dims+1
            self.task_lstm_network = MetaTaskLstmNetwork(
                input_size=args.lstm_input_size, lstm_hidden=lstm_hidden, num_lstm_layers=args.lstm_num_layers, lstm_out=0, device=self.device, use_softmax=args.use_softmax).to(self.device)
            self.task_lstm_optimizer = optim.Adam(
                self.task_lstm_network.parameters(), lr=self._lstm_lr)
            self.lstm_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.task_lstm_optimizer, T_max=args.num_train_iterations, eta_min=args.min_outer_lr)

        self.use_mlp_mean = args.use_mlp_mean

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

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.to(self.device)
            for name, param in params
            if param.requires_grad
        }

    def apply_inner_loop_update(self, loss, names_weights_copy, step, use_second_order=True):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """

        self.model.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    allow_unused=True, create_graph=use_second_order)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        # for key, grad in names_grads_copy.items():
        #     if grad is None:
        #         print('Grads not found for inner loop parameter', key)
        #     names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy, num_step=step)

        return names_weights_copy

    # zero_grad all meta parameters
    def zero_grad(self):
        """
        reset all gradients of meta paramters
        """
        # initialize meta parameters
        self.meta_optimizer.zero_grad()
        if self.use_adaptive_loss:
            self.loss_optimizer.zero_grad()
        if self.use_lstm:
            self.task_lstm_optimizer.zero_grad()
        if self.use_adaptive_loss_weight:
            self.task_info_optimizer.zero_grad()
        if self._use_learnable_params:
            self.lr_optimizer.zero_grad()

    def update_meta_params(self, mse_loss):
        """
        update all meta paramters
        :param mse_loss: meta mse loss
        """
        mse_loss.backward()
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=5.0)
        print(total_norm)

        self.meta_optimizer.step()
        self.meta_lr_scheduler.step()
        if self.use_adaptive_loss:
            self.loss_optimizer.step()
            # self.loss_lr_scheduler.step()
        if self.use_lstm:
            total_norm = 0.0
            for p in self.task_lstm_network.parameters():
                if p.requires_grad:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(total_norm)

            self.task_lstm_optimizer.step()
            # self.lstm_lr_scheduler.step()
        if self.use_adaptive_loss_weight:
            self.task_info_optimizer.step()
            # self.task_info_lr_scheduler.step()
        if self._use_learnable_params:
            total_norm = 0.0
            for p in self.inner_loop_optimizer.parameters():
                if p.requires_grad:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(total_norm)
            self.lr_optimizer.step()

        # forward on query data

    def query_forward(self, query_inputs, query_target_rating, names_weights_copy, mae_loss_fn, imp_weight=1):
        '''
        Forward propagation on query data
        Args:
            query_inputs: query inputs
            query_target_rating : query target rating
            names_weights_copy: inner loop parameters
            mae_loss_fn : mae loss function
            imp_weight : importance weight vector for gradients(multi step)
        return:
            query_loss : query loss containing gradients
            query_out_loss: query loss to show
            mae_loss: query mae loss to show
        '''
        # zero grad
        loss_fn = nn.MSELoss(reduction='none')

        # forward propagate on query data
        outputs = self.model(query_inputs, params=names_weights_copy)
        gt = torch.cat(
            (query_inputs[3], query_target_rating), dim=1)
        mask = (gt != 0)
        # compute loss
        if self.normalize_loss:
            query_loss = loss_fn(
                outputs*mask, gt*mask/5.0).sum()/mask.sum()
            mae_loss = mae_loss_fn(
                outputs[:, -1:].clone().detach()*5, query_target_rating)
            query_out_loss = torch.mean(loss_fn(
                outputs[:, -1:]*5.0, query_target_rating)).clone().detach().to("cpu")
        else:
            query_loss = loss_fn(outputs*mask, gt*mask).sum()/mask.sum()
            mae_loss = mae_loss_fn(
                outputs[:, -1:].clone().detach(), query_target_rating)
            query_out_loss = torch.mean(loss_fn(
                outputs[:, -1:], query_target_rating)).clone().detach().to("cpu")

        query_loss = query_loss * imp_weight

        return query_loss, query_out_loss, mae_loss

    def compute_adaptive_loss(self, loss, inputs, target_rating, step, mask, task_info):
        '''
        Compute Adaptive Loss
        Args:
            loss: base loss
            inputs: support data
            target_rating : target ratings
            step : current inner loop step
            task_info : task information of current task
        return:
            loss: adaptive loss
        '''

        if self.use_lstm:
            task_input = torch.cat(
                (inputs[3], target_rating), dim=1)
            task_info = self.task_lstm_network(
                task_input).squeeze()
            adapt_loss = loss * task_info * mask
            if self.use_mlp_mean:
                loss = self.loss_network(adapt_loss, step).squeeze()
                loss = torch.mean(loss)
            else:
                loss = adapt_loss.sum()/torch.count_nonzero(adapt_loss)

        else:
            # task_info_adapt = (task_info-task_info.mean()) / \
            #     (task_info.std() + 1e-12)
            task_info_adapt = torch.cat((task_info, loss.unsqueeze(2)), dim=2)
            if self.use_adaptive_loss_weight:
                weight = self.task_info_network(task_info_adapt)
                adapt_loss = weight * loss * mask
                if self.use_mlp_mean:
                    loss = self.loss_network(adapt_loss, step).squeeze()
                    loss = torch.mean(loss)
                else:
                    loss = adapt_loss.sum()/torch.count_nonzero(adapt_loss)
            else:
                loss = self.loss_network(
                    loss, step).squeeze()

        return loss

    # inner loop optimization
    def _inner_loop(self, support_data, task_info, query_inputs, query_target_rating, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            support_data: support data
            task_info: task information
            query_inputs: query data
            query_target_rating: query target
            train: if false, do not use multi step loss

        Returns:
            query_loss : query loss containing gradients
            query_out_loss: query loss to show
            mae_loss: query mae loss to show
        """

        # loss functions
        loss_fn = nn.MSELoss(reduction='none')
        mae_loss_fn = nn.L1Loss()

        task_mse_losses = []
        task_mse_out_losses = []
        task_mae_losses = []

        # inner loop parameters phi
        names_weights_copy = self.get_inner_loop_parameter_dict(
            self.model.named_parameters())
        # get importance weight
        imp_vecs = self.get_per_step_loss_importance_vector()

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
            outputs = self.model(inputs, params=names_weights_copy)
            gt = torch.cat(
                (inputs[3], target_rating), dim=1)
            mask = (gt != 0)
            # compute loss
            if self.normalize_loss:
                loss = loss_fn(outputs*mask, gt*mask/5.0)
            else:
                loss = loss_fn(outputs*mask, gt*mask)

            # adaptive loss
            if self.use_adaptive_loss:
                task_info_f = torch.cat(
                    (task_info, outputs.unsqueeze(2)), dim=2)
                loss = self.compute_adaptive_loss(
                    loss, inputs, target_rating, step, mask, task_info_f)

            else:
                loss = loss.sum()/mask.sum()

            # update inner loop paramters phi
            names_weights_copy = self.apply_inner_loop_update(
                loss=loss, names_weights_copy=names_weights_copy, use_second_order=True, step=step)

            ##### multi step loss - update meta paramters ######
            if self.use_multi_step and self._train_step < self.args.multi_step_loss_num_epochs and train:
                query_loss, query_out_loss, mae_loss = self.query_forward(
                    query_inputs, query_target_rating, names_weights_copy, mae_loss_fn, imp_vecs[step])
                task_mse_losses.append(query_loss)
                task_mse_out_losses.append(query_out_loss)
                task_mae_losses.append(mae_loss)

            ##### maml loss - update meta paramters at last step ####
            ### also use this step for valid set ###
            else:
                # at last step
                if step == self._num_inner_steps - 1:

                    query_loss, query_out_loss, mae_loss = self.query_forward(
                        query_inputs, query_target_rating, names_weights_copy, mae_loss_fn)
                    task_mse_losses.append(query_loss)
                    task_mse_out_losses.append(query_out_loss)
                    task_mae_losses.append(mae_loss)

        query_loss = torch.sum(torch.stack(task_mse_losses))
        query_out_loss = torch.mean(torch.stack(task_mse_out_losses))
        mae_loss = torch.mean(torch.stack(task_mae_losses))
        return query_loss, query_out_loss, mae_loss

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

        mse_loss_batch = []
        mse_loss_out_batch = []
        mae_loss_batch = []

        self.zero_grad()

        if train:
            self.model.train()
        else:
            self.model.eval()

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
            query_loss, query_out_loss, mae_loss = self._inner_loop(
                support, task_info, query_inputs, query_target_rating, train)  # do inner loop

            # collect loss data
            mse_loss_batch.append(query_loss)
            mse_loss_out_batch.append(query_out_loss)
            mae_loss_batch.append(mae_loss.detach().to("cpu").item())

        # set results
        mse_loss = torch.mean(torch.stack(mse_loss_batch))
        mse_loss_show = np.mean(mse_loss_out_batch)
        rmse_loss = np.sqrt(mse_loss_show)
        mae_loss = np.mean(mae_loss_batch)
        # Update meta parameters
        if train:
            self.update_meta_params(mse_loss)

        return mse_loss_show, rmse_loss, mae_loss

    def train(self, train_steps):
        wandb.init(project="MELO")
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
        wandb.config.update(self.args)

        val_batches = self.dataloader.generate_task(
            mode="valid", batch_size=self.val_size, normalized=self.normalize_loss, use_label=self.args.use_label)

        start_point = self._train_step+1
        # iteration
        for i in range(start_point, train_steps+1):
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
            if i % self.val_log_interval == 0:
                # set validation tasks
                val_mse_losses = []
                val_mae_losses = []
                for j in range(math.ceil(len(val_batches)/self.batch_size)):
                    mse_loss, _, mae_loss = self._outer_loop(
                        val_batches[j*self.batch_size: (j+1)*self.batch_size], train=False)
                    val_mse_losses.append(mse_loss)
                    val_mae_losses.append(mae_loss)

                mse_loss = np.mean(val_mse_losses)
                rmse_loss = np.sqrt(mse_loss)
                mae_loss = np.mean(val_mae_losses)

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
        test_mse_losses = []
        test_mae_losses = []
        for i in range(math.ceil(len(test_batches)/self.batch_size)):
            mse_loss, _, mae_loss = self._outer_loop(
                test_batches[i*self.batch_size: (i+1)*self.batch_size], train=False)
            test_mse_losses.append(mse_loss)
            test_mae_losses.append(mae_loss)

        mse_loss = np.mean(test_mse_losses)
        rmse_loss = np.sqrt(mse_loss)
        mae_loss = np.mean(test_mae_losses)

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
            f'Test RMSE loss: {rmse_loss:.4f} | '
            f'Test MAE loss: {mae_loss:.4f} | '
        )

    def load(self, checkpoint_step):
        '''
            load meta paramters
        '''
        target_path = os.path.join(
            self._save_dir, f"{self.args.model}_{checkpoint_step}_best_{self.args.mode}_{self.args.model}.pt")
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
            if self.use_lstm:
                self.task_lstm_network.load_state_dict(
                    checkpoint['lstm_model'])
            if self._use_learnable_params:
                self.inner_loop_optimizer.load_state_dict(
                    checkpoint['learning_rate']
                )

        except:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.')

    def _save_model(self):
        '''
            save meta paramters
        '''
        save_path = os.path.join(
            self._save_dir, f"{self.args.model}_{self._train_step}_best_{self.args.mode}_{self.args.model}.pt")
        model_dict = {
            'meta_model': self.model.state_dict()
        }
        if self.use_adaptive_loss:
            model_dict['loss_model'] = self.loss_network.state_dict()
        if self.use_adaptive_loss_weight:
            model_dict['loss_weight_model'] = self.task_info_network.state_dict()
        if self.use_lstm:
            model_dict['lstm_model'] = self.task_lstm_network.state_dict()
        if self._use_learnable_params:
            model_dict['learning_rate'] = self.inner_loop_optimizer.state_dict()
        torch.save(model_dict, save_path)

    def _load_pretrained_embedding(self):
        if torch.cuda.is_available():
            def map_location(storage, loc): return storage.cuda()
        else:
            map_location = 'cpu'

        if self.args.model == 'sasrec' or self.args.model == 'bert4rec':
            self.model.bert.bert_embedding.load_state_dict(torch.load(
                os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}_{self.args.bert_hidden_units}_{self.args.bert_num_blocks}_{self.args.bert_num_heads}"), map_location=map_location))
            # for param in self.model.bert.bert_embedding.parameters():
            # param.requires_grad = False
        else:
            self.model.embedding.load_state_dict(torch.load(
                os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}"), map_location=map_location))
            # for param in self.model.embedding.parameters():
            # param.requires_grad = False

    def _load_pretrained(self):
        if torch.cuda.is_available():
            def map_location(storage, loc): return storage.cuda()
        else:
            map_location = 'cpu'

        self.model.load_state_dict(torch.load(
            os.path.join(self._pretrained_dir, f"{self.args.model}_pretrained_{self.args.mode}_{self.args.bert_hidden_units}_{self.args.bert_num_blocks}_{self.args.bert_num_heads}"), map_location=map_location))


def main(args):
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.abspath('.'), "log/")

    print(f'log_dir: {args.log_dir}')

    maml = MAML(
        args
    )

    if args.checkpoint_step > -1:
        maml._train_step = args.checkpoint_step
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        if args.test_baseline:
            maml.test_baseline()
        else:
            maml.train(args.num_train_iterations)
    else:
        maml.test()


if __name__ == '__main__':
    main(args)
