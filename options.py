import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser('Train a MAML!')
# about log and data
parser.add_argument('--log_dir', type=str, default='./log',
                    help='directory to save to or load from')
parser.add_argument('--data_path', type=str, default='./Data/ml-1m/ratings.dat',
                    help='data path')
parser.add_argument('--mode', type=str, default='ml-1m',
                    help='ml-1m or ml-10m or amazon or yelp')
parser.add_argument('--test', default=False, action='store_true',
                    help='train or test')
parser.add_argument('--test_baseline', default=False, action='store_true',
                    help='test baseline method(using mean of ratings)')
parser.add_argument('--checkpoint_step', type=int, default=-1,
                    help=('checkpoint iteration to load for resuming '
                          'training, or for evaluation (-1 is ignored)'))
parser.add_argument('--log_interval', type=int, default=50,
                    help=('validation error logging interval'))
parser.add_argument('--num_test_data', type=int, default=1000,
                    help='the number of test data')
parser.add_argument('--min_sequence', type=int, default=20,
                    help='minimum number of reviews users should have')
parser.add_argument('--min_item', type=int, default=50,
                    help='minimum number of reviews items should have')
parser.add_argument('--random_seed', type=int, default=222,
                    help=('test data random seed'))

# hyperparmeters for training
parser.add_argument('--num_inner_steps', type=int, default=3,
                    help='number of inner-loop updates')
parser.add_argument('--use_learnable_params', type=boolean_string, default=True,
                    help='use learnable params or not learnable params')
parser.add_argument('--inner_lr', type=float, default=1e-3,
                    help='inner-loop learning rate initialization')
parser.add_argument('--outer_lr', type=float, default=1e-3,
                    help='outer-loop bert learning rate')
parser.add_argument('--min_outer_lr', type=float, default=1e-4,
                    help='min outer-loop bert learning rate')
parser.add_argument('--loss_lr', type=float, default=1e-3,
                    help='outer-loop learning rate')
parser.add_argument('--task_info_lr', type=float, default=1e-3,
                    help='outer-loop learning rate')
parser.add_argument('--lstm_lr', type=float, default=1e-3,
                    help='lstm learning rate')
parser.add_argument('--learn_lr', type=float, default=1e-3,
                    help='learning rate of learning rate')
parser.add_argument('--loss_weight_decay', type=float, default=0,
                    help='loss layer weight decay')
parser.add_argument('--num_train_iterations', type=int, default=2000,
                    help='number of outer-loop updates to train for')


# lstm model
parser.add_argument('--lstm_hidden', type=int, default=32,
                    help='the number of lstm hidden state dimension')
parser.add_argument('--lstm_num_layers', type=int, default=1,
                    help='the number of lstm layers')
parser.add_argument('--lstm_input_size', type=int, default=32,
                    help='input dimension of lstm')

# adaptive loss network
parser.add_argument('--loss_num_layers', type=int, default=1,
                    help='the number of adaptive loss layers')

# choice of model
parser.add_argument('--model', type=str, default='bert4rec',
                    help='model choice - sas4rec, bert4rec, narm, gru4rec')


# about transformer model
parser.add_argument('--max_seq_len', type=int, default=30,
                    help='maximum sequence length')
parser.add_argument('--bert_num_blocks', type=int, default=2,
                    help='number of bert blocks')
parser.add_argument('--bert_num_heads', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--bert_hidden_units', type=int, default=32,
                    help='number of hidden units ')
parser.add_argument('--bert_dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--model_init_seed', type=int, default=5,
                    help='init seed')

# narm model
parser.add_argument('--narm_hidden_size', type=int, default=32,
                    help='narm gru hidden size')
parser.add_argument('--narm_n_layers', type=int, default=1,
                    help='narm gru num layers')
parser.add_argument('--narm_embedding_dim', type=int, default=32,
                    help='narm embedding dimension')

# gru4rec model
parser.add_argument('--gru4rec_hidden_size', type=int, default=32,
                    help='gru4rec gru hidden size')
parser.add_argument('--gru4rec_n_layers', type=int, default=2,
                    help='gru4rec gru num layers')
parser.add_argument('--gru4rec_embedding_dim', type=int, default=32,
                    help='gru4rec embedding dimension')

# dataloader parameters
parser.add_argument('--num_users', type=int, default=0,
                    help='# of users')
parser.add_argument('--num_items', type=int, default=0,
                    help='number of items')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--val_size', type=int, default=600,
                    help='val batch size')
parser.add_argument('--num_samples', type=int, default=64,
                    help='number of subsamples')
parser.add_argument('--num_query_set', type=int, default=8,
                    help='number of query samples')
parser.add_argument('--default_rating', type=int, default=0,
                    help='rating value for padding')
parser.add_argument('--min_sub_window_size', type=int, default=15,
                    help=('minimum sequence during subsampling'))
parser.add_argument('--use_label', type=boolean_string, default=True,
                    help='use label as task information or input rating data as task information')

# training options
parser.add_argument('--multi_step_loss_num_epochs', type=int, default=1000,
                    help='number of epochs using multi step loss')
parser.add_argument('--use_multi_step', type=boolean_string, default=True,
                    help='use multi step loss or not')
parser.add_argument('--use_adaptive_loss', type=boolean_string, default=True,
                    help='use adaptive loss or pure maml')
parser.add_argument('--use_mlp', type=boolean_string, default=False,
                    help='use mlp adaptive loss')
parser.add_argument('--use_lstm', type=boolean_string, default=True,
                    help='use lstm for task information')
parser.add_argument('--use_shrinkage_loss', type=boolean_string, default=True,
                    help='use focal(shrinkage) loss')
parser.add_argument('--use_mlp_mean', type=boolean_string, default=True,
                    help='use lstm for task information')
parser.add_argument('--normalize_loss', type=boolean_string, default=True,
                    help='use normalized ratings')

# task information manipulation
parser.add_argument('--task_info_loss', type=boolean_string, default=True,
                    help='use loss as task information')
parser.add_argument('--task_info_rating_mean', type=boolean_string, default=True,
                    help='use mean rating as task information')
parser.add_argument('--task_info_rating_std', type=boolean_string, default=True,
                    help='use std of rating as task information')
parser.add_argument('--task_info_num_samples', type=boolean_string, default=True,
                    help='use the number of samples as task information')
parser.add_argument('--task_info_rating_distribution', type=boolean_string, default=True,
                    help='use the distribution of rating as task information')


# pretraining options
parser.add_argument('--pretraining_batch_size', type=int, default=128,
                    help='batch size during pretraining')
parser.add_argument('--pretrain_epochs', type=int, default=100,
                    help='the number of epochs for pretraining')
parser.add_argument('--pretraining_lr', type=float, default=1e-3,
                    help='learning rate during pretraining')
parser.add_argument('--load_pretrained_embedding', type=boolean_string, default=False,
                    help='load pretrained embeddding or not')
parser.add_argument('--load_pretrained', type=boolean_string, default=False,
                    help='load pretrained  or not')
parser.add_argument('--save_pretrained', type=boolean_string, default=False,
                    help='save embedding or whole model')
parser.add_argument('--pretrain_log_dir', type=str, default='./log_pretrained',
                    help='directory to save to or load from pretrained')

args = parser.parse_args()
