import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def change_padding_pos(ratings, device="cpu"):
    final_ratings = []
    lengths = []
    b, t = ratings.shape
    for rating in ratings:
        zero_len = (rating == 0).sum()
        lengths.append(int((t-zero_len).to("cpu").item()))
        final_ratings.append(torch.LongTensor(
            list(rating[zero_len:].long())+zero_len*[0]))
    final_ratings = torch.stack(final_ratings).to(device)
    input_lengths = torch.LongTensor(lengths)
    input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
    final_ratings = final_ratings[sorted_idx]
    return final_ratings, input_lengths


class MetaStepLossNetwork(nn.Module):
    def __init__(self, num_loss_hidden, num_loss_layers):
        super().__init__()
        # self.in_linear = nn.Linear(num_loss_hidden, num_loss_hidden, bias=True)
        # self.attention = nn.MultiheadAttention(1, 1, batch_first=True)
        self.layers = nn.ModuleList()
        for _ in range(num_loss_layers-1):
            self.layers.append(nn.Sequential(
                nn.Linear(num_loss_hidden, num_loss_hidden, bias=True),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(num_loss_hidden, 1, bias=True))

    def forward(self, x):
        # x = self.in_linear(x)
        # b, c = x.shape
        # x = x.reshape(b, c, 1)
        # x, _ = self.attention(x, x, x)
        # x = x.reshape(b, c)
        # for layer in self.layers:
        #     x = layer(x)
        return self.layers[-1](x)


class MetaLossNetwork(nn.Module):
    def __init__(self, num_inner_steps, num_loss_hidden, num_loss_layers):
        super().__init__()
        self.loss_layers = nn.ModuleList()
        for _ in range(num_inner_steps):
            self.loss_layers.append(MetaStepLossNetwork(
                num_loss_hidden, num_loss_layers))

    def forward(self, x, step):
        x = self.loss_layers[step](x)
        return x


class MetaTaskLstmNetwork(nn.Module):
    def __init__(self, input_size, lstm_hidden, num_lstm_layers, lstm_out=0, device="cpu"):
        super().__init__()
        if lstm_out == 0:
            lstm_out_size = lstm_hidden
        else:
            lstm_out_size = lstm_out
        self.embedding = nn.Embedding(7, input_size)
        self.h0 = nn.Parameter(torch.randn(num_lstm_layers,  lstm_out_size))
        self.c0 = nn.Parameter(torch.randn(num_lstm_layers,  lstm_hidden))
        self.lstm = nn.LSTM(
            batch_first=True, input_size=input_size, hidden_size=lstm_hidden, num_layers=num_lstm_layers, proj_size=lstm_out)
        self.device = device
        self.out_net = nn.Linear(lstm_out_size, 1)

    def forward(self, x):
        # x, lengths = change_padding_pos(x, self.device)
        x = x.long()
        x = self.embedding(x)
        b, t, _ = x.shape
        h0 = self.h0.repeat(b, 1, 1).permute(1, 0, 2).contiguous()
        c0 = self.c0.repeat(b, 1, 1).permute(1, 0, 2).contiguous()
        # embs = pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out, (hidden, c) = self.lstm(x, (h0, c0))
        # lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)

        return F.softmax(self.out_net(lstm_out).squeeze(), dim=0)


class MetaTaskMLPNetwork(nn.Module):
    def __init__(self, num_loss_weight_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_loss_weight_dims,
                      num_loss_weight_dims, bias=True),
            nn.ReLU(),
            nn.Linear(num_loss_weight_dims, 1, bias=True),
        )

    def forward(self, x):

        return F.softmax(self.mlp(x).squeeze(), dim=0)
