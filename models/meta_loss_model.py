import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MetaStepLossNetwork(nn.Module):
    def __init__(self, num_loss_hidden, num_loss_layers):
        super().__init__()
        # self.in_linear = nn.Linear(num_loss_hidden, num_loss_hidden, bias=True)
        # self.attention = nn.MultiheadAttention(1, 1, batch_first=True)
        self.layers = nn.ModuleList()
        for _ in range(num_loss_layers-1):
            self.layers.append(nn.Sequential(
                nn.Linear(num_loss_hidden, num_loss_hidden, bias=False),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(num_loss_hidden, 1, bias=False))

    def forward(self, x):
        # x = self.in_linear(x)
        # b, c = x.shape
        # x = x.reshape(b, c, 1)
        # x, _ = self.attention(x, x, x)
        # x = x.reshape(b, c)
        for layer in self.layers:
            x = layer(x)
        return x


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
    def __init__(self, input_size, lstm_hidden, num_lstm_layers, lstm_out=None):
        super().__init__()
        if lstm_out is None:
            lstm_out = lstm_hidden

        self.h0 = nn.Parameter(torch.randn(num_lstm_layers,  lstm_out))
        self.c0 = nn.Parameter(torch.randn(num_lstm_layers,  lstm_hidden))
        self.lstm = nn.LSTM(
            batch_first=True, input_size=input_size, hidden_size=lstm_hidden, num_layers=num_lstm_layers, proj_size=lstm_out)

    def forward(self, x):
        b, t, _ = x.shape
        h0 = self.h0.repeat(b, 1, 1).permute(1, 0, 2)
        c0 = self.c0.repeat(b, 1, 1).permute(1, 0, 2)
        return self.lstm(x, (h0, c0))
