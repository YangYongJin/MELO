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
import os
import numpy as np
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter


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

        # MAML++ multi-step updates
        self.use_multi_step = args.use_multi_step

        # use focal loss as inner loop loss function
        self.use_focal_loss = args.use_focal_loss

        # hyperparameters
        self._num_inner_steps = args.num_inner_steps
        self._inner_lr = args.inner_lr
        self._outer_lr = args.outer_lr

        # normalize user ratings range from 0 to 1
        self.normalize_loss = args.normalize_loss

        # inner loop optimizer - use learnable inner loop learning rates
        self._use_learnable_params = args.use_learnable_params
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=self.device, total_num_inner_loop_steps=self._num_inner_steps, use_learnable_learning_rates=self._use_learnable_params, init_learning_rate=self._inner_lr)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.model.named_parameters()))

        # optimizer for inner loop lr
        if self._use_learnable_params:
            self._learning_lr = args.learn_lr
            self.lr_optimizer = optim.Adam(
                self.inner_loop_optimizer.parameters(), lr=self._learning_lr)

        # meta model optimizer
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), lr=self._outer_lr)

        # meta learning rate scheduler (cosine annealing scheduler)
        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=args.num_train_iterations, eta_min=args.min_outer_lr)

        # current epoch
        self._train_step = 0

        # Options for Adaptive Wighted Loss
        self.use_adaptive_loss = args.use_adaptive_loss
        self.use_adaptive_loss_weight = (
            args.use_mlp and self.use_adaptive_loss)
        self.use_lstm = (
            args.use_lstm and self.use_adaptive_loss)
        num_loss_dims = None

        # mlp mean loss network - aggregate item losses using mlp layers
        if self.use_adaptive_loss:
            self._loss_lr = args.loss_lr
            num_loss_dims = args.max_seq_len
            self.loss_network = MetaLossNetwork(
                self._num_inner_steps, num_loss_dims, args.loss_num_layers, use_step_loss=args.use_step_loss).to(self.device)
            self.loss_optimizer = optim.Adam(
                self.loss_network.parameters(), lr=self._loss_lr, weight_decay=args.loss_weight_decay)
            self.loss_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.loss_optimizer, milestones=[
                            500, 1000, 1500], gamma=0.7)

        # STATS network
        # loss, mean, std, labels, and predictions are included for statistical information
        self.task_info_predictions = args.task_info_predictions
        self.task_info_loss = args.task_info_loss
        if self.use_adaptive_loss_weight:
            num_loss_weight_dims = 1*args.task_info_loss + 1*args.task_info_rating_mean+1 * \
                args.task_info_rating_std+1*args.task_info_predictions+1*args.task_info_labels
            self._task_info_lr = args.task_info_lr
            self.task_info_network = MetaTaskMLPNetwork(
                num_loss_weight_dims, use_softmax=args.use_softmax).to(self.device)
            self.task_info_optimizer = optim.Adam(
                self.task_info_network.parameters(), lr=self._task_info_lr)
            self.task_info_lr_scheduler = optim.lr_scheduler.\
                MultiStepLR(self.task_info_optimizer, milestones=[
                            500, 1000, 1500], gamma=0.7)

        # lstm loss network
        if self.use_lstm:
            self._lstm_lr = args.lstm_lr
            lstm_hidden = args.lstm_hidden
            self.task_lstm_network = MetaTaskLstmNetwork(
                input_size=args.lstm_input_size, lstm_hidden=lstm_hidden, num_lstm_layers=args.lstm_num_layers, lstm_out=0, device=self.device, use_softmax=args.use_softmax).to(self.device)
            self.task_lstm_optimizer = optim.Adam(
                self.task_lstm_network.parameters(), lr=self._lstm_lr)
            self.lstm_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.task_lstm_optimizer, T_max=args.num_train_iterations, eta_min=1e-2)

        self.use_mlp_mean = args.use_mlp_mean

        self.rating_info = {}
        for i in range(1,6):
            self.rating_info['rating_'+str(i)] = {}
            self.rating_info['rating_'+str(i)]['loss'] = []
            self.rating_info['rating_'+str(i)]['pred'] = []
            self.rating_info['rating_'+str(i)]['num'] = []

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
        :param step: Current step's index.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :return: A dictionary with the updated weights (name, param)
        """

        self.model.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    allow_unused=True, create_graph=use_second_order)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

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
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=5.0)

        self.meta_optimizer.step()
        self.meta_lr_scheduler.step()
        if self.use_adaptive_loss and self.use_mlp_mean:
            self.loss_optimizer.step()
        if self.use_lstm:

            self.task_lstm_optimizer.step()
            self.lstm_lr_scheduler.step()
        if self.use_adaptive_loss_weight:
            self.task_info_optimizer.step()
            # self.task_info_lr_scheduler.step()
        if self._use_learnable_params:
            self.lr_optimizer.step()

        # forward on query data

    def eval_by_rating(self, output, target_rating, loss_fn):
        with torch.no_grad():
            for i in range(1,6):
                if (target_rating == i).sum() > 0:
                    rating_value = (torch.sum(loss_fn(
                            output*(target_rating == i), target_rating*(target_rating == i)))/(target_rating == i).sum())
                    if (i==1):
                        print(i, loss_fn(
                                output*(target_rating == i), target_rating*(target_rating == i)))
                        print(torch.sum(loss_fn(
                            output*(target_rating == i), target_rating*(target_rating == i)))/(target_rating == i).sum())
                
                    self.rating_info['rating_'+str(i)]['loss'].append(rating_value.item())
                    self.rating_info['rating_'+str(i)]['pred'] += (output[target_rating == i]).tolist()
                    self.rating_info['rating_'+str(i)]['num'].append((target_rating == i).sum().item())


    def query_forward(self, query_inputs, query_target_rating, names_weights_copy, mae_loss_fn, imp_weight=1, train=False):
        '''
        Forward propagation on query data
        Args:
            query_inputs: query set inputs
            query_target_rating : query set target rating
            names_weights_copy: inner loop parameters
            mae_loss_fn : mae loss function
            imp_weight : importance weight vector for gradients(multi step loss)
        return:
            query_loss : query loss containing gradients
            query_out_loss: query loss to visualize
            mae_loss: query mae loss to visualize
        '''
        # zero grad
        loss_fn = nn.MSELoss(reduction='none')

        # forward propagate on query data
        outputs = self.model(query_inputs, params=names_weights_copy)

        # labels
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
            if not train:
                self.eval_by_rating(outputs[:, -1:]*5.0, query_target_rating, loss_fn)
        else:
            query_loss = loss_fn(outputs*mask, gt*mask).sum()/mask.sum()
            mae_loss = mae_loss_fn(
                outputs[:, -1:].clone().detach(), query_target_rating)
            query_out_loss = torch.mean(loss_fn(
                outputs[:, -1:], query_target_rating)).clone().detach().to("cpu")
            if not train:
                self.eval_by_rating(outputs[:, -1:], query_target_rating, loss_fn)
        

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
            mask : mask for padded items
            task_info : task information of current task(e.g. mean, std)
        return:
            loss: adaptive loss
        '''

        # use lstm state encoder
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
            # use mlp state encoder
            if self.task_info_loss:
                task_info_adapt = torch.cat(
                    (task_info, loss.unsqueeze(2)), dim=2)
            else:
                task_info_adapt = task_info
            # task_info_adapt = (task_info-task_info.mean()) / \
            #     (task_info.std() + 1e-12)
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

    def focal_loss(self, x, y, ord=3):
        """
            focal loss for regression 
        """
        return torch.pow(torch.abs(y-x), ord)

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
        mae_loss_fn = nn.L1Loss()

        # option for focal loss
        if self.use_focal_loss:
            loss_fn = self.focal_loss
        else:
            loss_fn = nn.MSELoss(reduction='none')

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
            # compute mse loss
            if self.normalize_loss:
                loss = loss_fn(outputs*mask, gt*mask/5.0)
            else:
                loss = loss_fn(outputs*mask, gt*mask)

            # adaptive weighted loss
            if self.use_adaptive_loss:
                if self.task_info_predictions:
                    task_info_f = torch.cat(
                        (task_info, outputs.unsqueeze(2)), dim=2)
                else:
                    task_info_f = task_info
                loss = self.compute_adaptive_loss(
                    loss, inputs, target_rating, step, mask, task_info_f)

            # normal mse loss
            else:
                loss = loss.sum()/mask.sum()

            # update inner loop paramters phi
            names_weights_copy = self.apply_inner_loop_update(
                loss=loss, names_weights_copy=names_weights_copy, use_second_order=True, step=step)

            ##### multi step loss - update meta paramters ######
            if self.use_multi_step and self._train_step < self.args.multi_step_loss_num_epochs and train:
                query_loss, query_out_loss, mae_loss = self.query_forward(
                    query_inputs, query_target_rating, names_weights_copy, mae_loss_fn, imp_vecs[step], train)
                task_mse_losses.append(query_loss)
                task_mse_out_losses.append(query_out_loss)
                task_mae_losses.append(mae_loss)

            ##### maml loss - update meta paramters at last step ####
            ### also use this step for valid set ###
            else:
                # at last step
                if step == self._num_inner_steps - 1:

                    query_loss, query_out_loss, mae_loss = self.query_forward(
                        query_inputs, query_target_rating, names_weights_copy, mae_loss_fn, train)
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
        """Train the MAML.

        Optimizes MAML meta-parameters
        while periodically validating on validation_tasks, logging metrics, and
        saving checkpoints.

        Args:
            train_steps (int) : the number of steps this model should train for
        """
        print(f"Starting MAML training at iteration {self._train_step}")

        # initialize wandb project
        if self.use_adaptive_loss:
            wandb.init(
                project=f"MELO-TRAIN-{self.args.model}-{self.args.mode}")
        else:
            wandb.init(
                project=f"MAML-TRAIN-{self.args.model}-{self.args.mode}")

        # define tensorboard writer and wandb config
        # # writer = SummaryWriter(log_dir=self._log_dir)
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
                # writer.add_scalar("train/MSEloss", mse_loss, self._train_step)
                # writer.add_scalar("train/RMSEloss",
                                #   rmse_loss, self._train_step)
                # writer.add_scalar("train/MAEloss", mae_loss, self._train_step)

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
                self._save_model(best=False)
                # Save the best model wrt valid rmse loss
                if self.best_valid_rmse_loss > rmse_loss:
                    self.best_valid_rmse_loss = rmse_loss
                    self.best_step = i
                    self._save_model()
                    print(
                        f'........Model saved (step: {self.best_step} | RMSE loss: {rmse_loss:.4f})')

                # writer.add_scalar("valid/MSEloss", mse_loss, self._train_step)
                # writer.add_scalar("valid/RMSEloss",
                                #   rmse_loss, self._train_step)
                # writer.add_scalar("valid/MAEloss", mae_loss, self._train_step)
        # writer.close()

        print("-------------------------------------------------")
        print("Model with the best validation RMSE loss is saved.")
        print(f'Best step: {self.best_step}')
        print(f'Best RMSE loss: {self.best_valid_rmse_loss:.4f}')
        print("Done.")

    def test(self):
        '''
            Test on test batches
        '''

        # initialize wandb project
        if self.use_adaptive_loss:
            wandb.init(project=f"MELO-TEST-{self.args.model}-{self.args.mode}")
        else:
            wandb.init(project=f"MAML-TEST-{self.args.model}-{self.args.mode}")

        # define wandb config
        wandb.config.update(self.args)

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
            f'Test RMSE loss: {rmse_loss:.4f} | '
            f'Test MAE loss: {mae_loss:.4f} | '
        )
        print(' -------- Rating ---- ')
        for k,v in self.rating_info.items():
            print('Information of ', k)
            print('The Number of items', np.sum(v['num']))
            print('Loss Mean', np.sqrt(np.mean((v['loss']))))
            print('Prediction Mean', np.mean((v['pred'])))
            print('Prediction Median', np.median((v['pred'])))
            print('Prediction Std', np.std((v['pred'])))
        wandb.log({
            "Test RMSE loss": rmse_loss,
            "Test MAE loss": mae_loss
        })

    def test_baseline(self):
        '''
            Test baseline(using mean)
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

    def load(self, checkpoint_step, best=True):
        '''
            load meta paramters
        '''
        if best:
            target_path = os.path.join(
                self._save_dir, f"{self.args.model}_{checkpoint_step}_best_{self.args.mode}_{self.args.model}.pt")
        else:
            target_path = os.path.join(
                self._save_dir, f"{self.args.model}_{checkpoint_step}_{self.args.mode}_{self.args.model}.pt")
        print("Loading checkpoint from", target_path)
        try:
            if torch.cuda.is_available():
                def map_location(storage, loc): return storage.cuda()
            else:
                map_location = 'cpu'
            checkpoint = torch.load(target_path, map_location=map_location)
            self.model.load_state_dict(checkpoint['meta_model'])
            self.meta_lr_scheduler.load_state_dict(
                checkpoint['meta_model_scheduler'])
            self.meta_optimizer.load_state_dict(
                checkpoint['meta_model_optimizer'])
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

    def _save_model(self, best=True):
        '''
            save meta paramters
        '''
        if best:
            save_path = os.path.join(
                self._save_dir, f"{self.args.model}_{self._train_step}_best_{self.args.mode}_{self.args.model}.pt")
        else:
            save_path = os.path.join(
                self._save_dir, f"{self.args.model}_{self._train_step}_{self.args.mode}_{self.args.model}.pt")
        model_dict = {
            'meta_model': self.model.state_dict(),
            'meta_model_scheduler': self.meta_lr_scheduler.state_dict(),
            'meta_model_optimizer': self.meta_optimizer.state_dict()
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
        """
            load embedding parts of meta model
        """
        if torch.cuda.is_available():
            def map_location(storage, loc): return storage.cuda()
        else:
            map_location = 'cpu'

        if self.args.model == 'sasrec' or self.args.model == 'bert4rec':
            self.model.bert.bert_embedding.load_state_dict(torch.load(
                os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}_{self.args.bert_hidden_units}_{self.args.bert_num_blocks}_{self.args.bert_num_heads}"), map_location=map_location))

        else:
            self.model.embedding.load_state_dict(torch.load(
                os.path.join(self._embedding_dir, f"{self.args.model}_embedding_{self.args.mode}"), map_location=map_location))

    def _load_pretrained(self):
        """
            load pretrained meta model (all parameters of model)
        """
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
        maml.load(args.checkpoint_step, args.test_best)
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
