import torch
import torch.nn as nn


class MetaLossNetwork(nn.Module):
    def __init__(self, num_loss_hidden, num_loss_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_loss_layers-1):
            self.layers.append(nn.Sequential(
                nn.Linear(num_loss_hidden, num_loss_hidden, bias=False),
                nn.ReLU()
            ))
        self.layers.append(nn.Linear(num_loss_hidden, 1, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MetaTaskLstmNetwork(nn.Module):
    def __init__(self, input_size, lstm_hidden, num_lstm_layers):
        super().__init__()
        self.h0 = nn.Parameter(torch.randn(1,  9))
        self.c0 = nn.Parameter(torch.randn(1,  9))
        self.lstm = nn.LSTM(
            batch_first=True, input_size=input_size, hidden_size=lstm_hidden, num_layers=num_lstm_layers)

    def forward(self, x):
        b, t, _ = x.shape
        h0 = self.h0.repeat(1, b, 1)
        c0 = self.c0.repeat(1, b, 1)
        return self.lstm(x, (h0, c0))
