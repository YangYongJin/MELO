import torch
import torch.nn as nn
from .base import MetaLinearLayer, MetaBERTEmbedding, extract_top_level_dict, MetaGRUModel


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
            vocab_size=vocab_size,  embed_size=self.embedding_dim, max_len=max_len, dropout=dropout, needs_position=False)
        self.gru = MetaGRUModel(
            self.embedding_dim, self.hidden_size, self.n_layers, self.hidden_size)
        self.a_1 = MetaLinearLayer(
            self.hidden_size, self.hidden_size, use_bias=False)
        self.a_2 = MetaLinearLayer(
            self.hidden_size, self.hidden_size, use_bias=False)
        self.v_t = MetaLinearLayer(self.hidden_size, 1, use_bias=False)
        self.ct_dropout = nn.Dropout(dropout)

        self.out_layer = MetaLinearLayer(self.hidden_size, 1, use_bias=True)

    def forward(self, inputs, params=None):

        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)
            embedding_params = param_dict['embedding']
            gru_params = param_dict['gru']
            a_1_params = param_dict['a_1']
            a_2_params = param_dict['a_2']
            v_t_params = param_dict['v_t']
            out_params = param_dict['out_layer']

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
        c_local = alpha.unsqueeze(2).expand_as(gru_out) * gru_out

        # c_t = torch.cat([c_local, c_global], 1)
        # print(c_t.shape)
        c_t = c_global.unsqueeze(1)*c_local

        c_t = self.ct_dropout(c_t)

        out = self.out_layer(c_t, params=out_params)

        b, t, d = out.shape
        out = out.view(b, -1)

        return 0.1 + torch.sigmoid(out)

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
