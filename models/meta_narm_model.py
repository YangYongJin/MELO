import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict


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


class MetaGRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, use_bias=True):
        super(MetaGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = MetaLinearLayer(input_size, 3 * hidden_size, use_bias)
        self.h2h = MetaLinearLayer(hidden_size, 3 * hidden_size, use_bias)

    def forward(self, x, hidden, params=None):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            x2h_params = params['x2h']
            h2h_params = params['h2h']
        else:
            x2h_params = None
            h2h_params = None
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x, params=x2h_params)
        gate_h = self.h2h(hidden, params=h2h_params)
        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class MetaGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bias=True):
        super(MetaGRUModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.layer_dict = nn.ModuleDict()

        self.layer_dict['gru0'] = MetaGRUCell(self.input_size,
                                              self.hidden_size,
                                              self.bias)
        for l in range(1, self.num_layers):
            self.layer_dict['gru{0}'.format(l)] = MetaGRUCell(
                self.hidden_size, self.hidden_size, self.bias)
        self.layer_dict['fc'] = MetaLinearLayer(hidden_size, output_size)

        self.h0 = nn.Parameter(torch.randn(num_layers, hidden_size))

    def forward(self, x, params=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        param_dict = {}
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            h0 = param_dict['h0']
        else:
            h0 = self.h0

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        outs = []

        b, _, _ = x.shape
        h0 = h0.repeat(b, 1, 1).permute(1, 0, 2)

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(x.size(1)):

            for layer in range(self.num_layers):
                layer_name = 'gru{0}'.format(layer)
                if layer == 0:
                    hidden_l = self.layer_dict[layer_name](
                        x[:, t, :], hidden[layer], params=param_dict[layer_name])
                else:
                    hidden_l = self.layer_dict[layer_name](
                        hidden[layer - 1], hidden[layer], params=param_dict[layer_name])
                hidden[layer] = hidden_l

            outs.append(hidden_l)

        out = torch.stack(outs).permute(1, 0, 2)
        # Take only last time step. Modify for seq to seq
        # out = outs[-1].squeeze()
        h_n = outs[-1]
        out = self.layer_dict['fc'](out, params=param_dict['fc'])

        return out, h_n


class MetaNARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class
    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers
    """

    def __init__(self, args):
        args.device = "cpu"
        super(MetaNARM, self).__init__()
        self.hidden_size = args.narm_hidden_size
        self.n_layers = args.narm_n_layers
        max_len = args.max_seq_len-1
        dropout = args.bert_dropout
        num_items = args.num_items
        vocab_size = num_items + 2
        self.device = args.device
        self.embedding_dim = args.narm_embedding_dim
        self.embedding = MetaBERTEmbedding(
            vocab_size=vocab_size,  embed_size=self.embedding_dim, max_len=max_len, dropout=dropout)
        self.gru = MetaGRUModel(
            self.embedding_dim, self.hidden_size, self.n_layers, self.hidden_size)
        self.a_1 = MetaLinearLayer(
            self.hidden_size, self.hidden_size, use_bias=False)
        self.a_2 = MetaLinearLayer(
            self.hidden_size, self.hidden_size, use_bias=False)
        self.v_t = MetaLinearLayer(self.hidden_size, 1, use_bias=False)

        self.out = MetaLinearLayer(2*self.hidden_size, 1, use_bias=True)

    def forward(self, inputs, params=None):

        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
            embedding_params = param_dict['embedding']
            gru_params = param_dict['gru']
            a_1_params = param_dict['a_1']
            a_2_params = param_dict['a_2']
            v_t_params = param_dict['v_t']
            out_params = param_dict['out']

        else:
            embedding_params = None
            gru_params = None
            a_1_params = None
            a_2_params = None
            v_t_params = None
            out_params = None

        x = self.embedding(inputs, params=embedding_params)
        gru_out, h_n = self.gru(x, params=gru_params)

        # fetch the last hidden state of last timestamp
        ht = h_n  # b * h
        gru_out = gru_out  # .permute(1, 0, 2) # b* t*h

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1,
                      self.hidden_size), params=a_1_params).view(gru_out.size())
        q2 = self.a_2(ht, params=a_2_params)

        mask = torch.where(torch.cat((inputs[1], inputs[2]), dim=1) > 0, torch.tensor(
            [1.], device=self.device), torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)

        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1,
                         self.hidden_size), params=v_t_params).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        # print(c_local.shape)

        c_t = torch.cat([c_local, c_global], 1)
        # c_t = self.ct_dropout(c_t)

        out = self.out(c_t, params=out_params)

        return out

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


inputs = (torch.LongTensor(np.ones((1, 1))), torch.LongTensor(np.ones((1, 29))),
          torch.LongTensor(np.ones((1, 1))), torch.FloatTensor(np.ones((1, 29))))

args = EasyDict({
    'narm_hidden_size': 64,
    'narm_n_layers': 2,
    'max_seq_len': 30,
    'bert_dropout': 0.3,
    'num_items': 500,
    'narm_embedding_dim': 16,
    'narm_output_dim': 32})

narm = MetaNARM(args)
out = narm(inputs)
