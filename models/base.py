import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

############### linear layer ###############


class MetaLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_features: The shape of the input features
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


############## embedding ##################################
class MetaEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
            A MetaEmbedding layer. Applies the same functionality of a standard embedding layer with the added functionality o
            being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
            weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
            learning setting.
            :param num_embeddings: The shape of the input vocab size
            :param embedding_dim: size of embedding dimensions
        """
        super(MetaEmbedding, self).__init__()

        self.weights = nn.Parameter(torch.ones(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, params=None):
        """
        Embedding Operation. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, l, d)
        :param params: A dictionary containing 'embedding weights'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the embedding function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
        else:
            weight = self.weights
        return F.embedding(input=x, weight=weight)


class MetaPositionalEmbedding(nn.Module):
    """
        Meta Positional Embedding Layer. It does similar operation as MetaEmbedding Layer.
    """

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.weights = nn.Parameter(torch.ones(max_len+1, d_model))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
        else:
            weight = self.weights
        batch_size = x.size(0)
        return weight.unsqueeze(0).repeat(batch_size, 1, 1)


class MetaBERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, needs_position=True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param max_len: maximum sequence length
        :param dropout: dropout rate
        :param needs_position: True if positional embedding is needed
        """
        super().__init__()
        self.embedding = MetaEmbedding(
            num_embeddings=vocab_size, embedding_dim=embed_size)
        self.needs_position = needs_position
        if self.needs_position:
            self.position = MetaPositionalEmbedding(
                max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, params=None):
        user_id, product_history, target_product_id,  product_history_ratings = inputs

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            embedding_params = params['embedding']
            if self.needs_position:
                position_params = params['position']
        else:
            embedding_params = None
            if self.needs_position:
                position_params = None
        product_his = torch.cat((product_history, target_product_id), dim=1)
        x = self.embedding(product_his, params=embedding_params)

        if self.needs_position:
            x += self.position(product_his, params=position_params)
        B, T = product_history_ratings.shape

        return self.dropout(x)


########################## GRU #################
class MetaGRUCell(nn.Module):

    """
    An implementation of GRUCell for meta learning setting.
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

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class MetaGRUModel(nn.Module):
    """
    An implementation of   GRU for meta learning setting.
    """

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


################### gelu and layer norm #######################################
class GELU(nn.Module):
    """
    GELU non linear activation.
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MetaLayerNorm(nn.Module):
    "Construct a layernorm module (See citation at original paper for details). "

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


################### positionwise and feed forward ############################
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
