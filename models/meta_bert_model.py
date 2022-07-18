import numbers
from copy import copy
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = {}
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level in output_dict:
            new_item = {key: value for key,
                        value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

        elif sub_level == "":
            output_dict[top_level] = current_dict[key]
        else:
            output_dict[top_level] = {sub_level: current_dict[key]}
    # print(current_dict.keys(), output_dict.keys())
    return output_dict


class MetaLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param out_features: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        elif self.use_bias:
            weight, bias = self.weights, self.bias
        else:
            weight = self.weights
            bias = None
        return F.linear(input=x, weight=weight, bias=bias)


class MetaEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
            A MetaEmbedding layer. Applies the same functionality of a standard embedding layer with the added functionality o
            being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
            weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
            learning setting.
            :param input_shape: The shape of the input data, in the form (b, f)
            :param num_filters: Number of output filters
            :param use_bias: Whether to use biases or not.
        """
        super(MetaEmbedding, self).__init__()

        self.weights = nn.Parameter(torch.ones(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
        else:
            weight = self.weights
        return F.embedding(input=x, weight=weight)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MetaLayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(MetaLayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x, params=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            a_2 = params["a_2"]
            b_2 = params["b_2"]
        else:
            a_2 = self.a_2
            b_2 = self.b_2
        return a_2 * (x - mean) / (std + self.eps) + b_2


class MetaPositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.weights = nn.Parameter(torch.ones(max_len, d_model))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
        else:
            weight = self.weights
        batch_size = x.size(0)
        return weight.unsqueeze(0).repeat(batch_size, 1, 1)


########### ---- not basic fns --------- ############
class MetaBERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.embedding = MetaEmbedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)

        self.position = MetaPositionalEmbedding(
            max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, params=None):
        user_id, product_history, target_product_id,  product_history_ratings = inputs

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            embedding_params = params['embedding']
            position_params = params['position']
        else:
            embedding_params = None
            position_params = None

        x = self.embedding(product_history, params=embedding_params) + \
            self.position(product_history, params=position_params)
        B, T = product_history_ratings.shape

        target_info = self.embedding(
            target_product_id, params=embedding_params).view(B, 1, -1)
        x = x*product_history_ratings.view(B, T, 1)
        x = torch.cat([x, target_info], dim=1)
        return self.dropout(x)


### attention module ###########
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

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


class MetaSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(MetaSublayerConnection, self).__init__()
        self.norm = MetaLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, params=None, sub_params=None):
        "Apply residual connection to any sublayer with the same size."
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            norm_params = params['norm']
        else:
            norm_params = None

        if sub_params is not None:
            return x + self.dropout(sublayer(self.norm(x, params=norm_params), params=sub_params))
        else:
            return x + self.dropout(sublayer(self.norm(x, params=norm_params)))


class MetaPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MetaPositionwiseFeedForward, self).__init__()
        self.linear1 = MetaLinearLayer(
            in_features=d_model, out_features=d_ff, use_bias=True)
        self.linear2 = MetaLinearLayer(
            in_features=d_ff, out_features=d_model, use_bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            linear1_params = params['linear1']
            linear2_params = params['linear2']
        else:
            linear1_params = None
            linear2_params = None
        return self.linear2(self.dropout(self.activation(self.linear1(x, params=linear1_params))), params=linear2_params)


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
        # (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = None
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


class MetaLossNetwork(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.linear1 = MetaLinearLayer(
            1, 3, use_bias=False)
        self.linear2 = MetaLinearLayer(
            3, 1, use_bias=False)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            linear1_params = params['linear1']
            linear2_params = params['linear2']

        else:
            linear1_params = None
            linear2_params = None
        x = self.linear1(x, params=linear1_params)
        x = self.linear2(x, params=linear2_params)
        return x
