
import math

import torch.nn as nn
import torch.nn.functional as F
import torch

from .base import extract_top_level_dict,  MetaBERTEmbedding, MetaLinearLayer, MetaSublayerConnection, MetaPositionwiseFeedForward


### attention module ###########
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MetaMultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.layer_dict = nn.ModuleDict()

        for i in range(3):
            self.layer_dict['in_linear{}'.format(i)] = MetaLinearLayer(
                in_features=d_model, out_features=d_model, use_bias=True)
        self.layer_dict['out_linear'] = MetaLinearLayer(
            in_features=d_model, out_features=d_model, use_bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, params=None):
        param_dict = {}
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        lst = []
        for i, x in enumerate([query, key, value]):
            lst.append(self.layer_dict['in_linear{}'.format(i)](x, params=param_dict['in_linear{}'.format(
                i)]).view(batch_size, -1, self.h, self.d_k).transpose(1, 2))

        query, key, value = lst

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        return self.layer_dict['out_linear'](x, params=param_dict['out_linear'])


class MetaTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MetaMultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = MetaPositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = MetaSublayerConnection(
            size=hidden, dropout=dropout)
        self.output_sublayer = MetaSublayerConnection(
            size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            attention_params = params['attention']
            feed_forward_params = params['feed_forward']
            input_sublayer_params = params['input_sublayer']
            output_sublayer_params = params['output_sublayer']
        else:
            attention_params = None
            feed_forward_params = None
            input_sublayer_params = None
            output_sublayer_params = None
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, params=attention_params), params=input_sublayer_params)
        x = self.output_sublayer(
            x, self.feed_forward, params=output_sublayer_params, sub_params=feed_forward_params)
        return self.dropout(x)

## bert ##


class MetaBERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        # fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.max_seq_len-1
        num_items = args.num_items
        # num_users = args.num_users
        self.n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
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
        for i in range(self.n_layers):
            self.layer_dict['transformer{}'.format(i)] = MetaTransformerBlock(
                hidden, heads, hidden * 4, dropout)

    def forward(self, inputs, params=None):
        x = torch.cat((inputs[1], inputs[2]), dim=1)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # mask = None
        param_dict = {}
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
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
            x = self.layer_dict['transformer{}'.format(i)].forward(
                x, mask, params=param_dict['transformer{}'.format(i)])
        return x


class MetaBERT4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = MetaBERT(args)
        self.dim_reduct = MetaLinearLayer(self.bert.hidden, 16)
        self.out1 = MetaLinearLayer(16*(args.max_seq_len), 128)
        self.out2 = MetaLinearLayer(128, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs, params=None):
        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
            bert_params = param_dict['bert']
            dim_reduct_params = param_dict['dim_reduct']
            out1_params = param_dict['out1']
            out2_params = param_dict['out2']

        else:
            bert_params = None
            dim_reduct_params = None
            out1_params = None
            out2_params = None

        x = self.bert(inputs, params=bert_params)
        x = self.dim_reduct(x, params=dim_reduct_params)
        b, t, d = x.shape
        x = x.view(b, -1)  # Batch size x (t*d)
        x = self.relu(x)
        x = self.out1(x, params=out1_params)
        x = self.relu(x)
        x = self.out2(x, params=out2_params)
        return 0.1 + torch.sigmoid(x)

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
