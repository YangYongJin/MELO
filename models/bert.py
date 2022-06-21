from .bert_modules.bert import BERT

import torch.nn as nn
import torch


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BERT(args)
        self.dim_reduct = nn.Linear(self.bert.hidden, 16)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16*(args.max_seq_len), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, inputs):
        x = self.bert(inputs)
        x = self.dim_reduct(x)
        b, t, d = x.shape
        x = x.view(b, -1)  # Batch size x (t*d)
        return 0.1 + torch.sigmoid(self.out(x))
