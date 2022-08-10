

import torch.nn as nn
import torch
import math

from .base import extract_top_level_dict,  MetaBERTEmbedding, MetaLinearLayer


## bert ##
class MetaNCF(nn.Module):
    def __init__(self, args):
        super().__init__()

        # fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.max_seq_len-1
        num_items = args.num_items
        vocab_size = num_items + 2
        # user_vocab_size = num_users + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        self.layer_dict = nn.ModuleDict()

        # embedding for BERT, sum of positional, segment, token embeddings
        self.bert_embedding = MetaBERTEmbedding(
            vocab_size=vocab_size,  embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        cur_layer = hidden
        self.n_layers = int(math.log(cur_layer, 2))
        for i in range(self.n_layers):
            self.layer_dict['linear{}'.format(
                i)] = MetaLinearLayer(cur_layer, cur_layer//2)
            cur_layer = cur_layer // 2

    def forward(self, inputs, params=None):
        param_dict = {}
        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
            if 'bert_embedding' not in param_dict.keys():
                bert_embedding_params = None
            else:
                bert_embedding_params = param_dict['bert_embedding']
        else:
            bert_embedding_params = None

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # embedding the indexed sequence to sequence of vectors
        x = self.bert_embedding(inputs, params=bert_embedding_params)

        # running over multiple transformer blocks
        for i in range(self.n_layers):
            x = self.layer_dict['linear{}'.format(i)].forward(
                x, params=param_dict['linear{}'.format(i)])
        return x.squeeze()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    # print(param.grad)
                    param.grad.zero_()
        else:
            for name, param in params.items():
                if (
                    param.requires_grad == True
                    and param.grad is not None
                    and torch.sum(param.grad) > 0
                ):
                    # print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None
