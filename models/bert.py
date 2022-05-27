from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BERT(args)
        self.dim_reduct = nn.Linear(self.bert.hidden, 32)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32*(args.seq_len+2), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        x = self.bert(inputs)
        x = self.dim_reduct(x)
        b, t, d = x.shape
        x = x.view(b, -1)
        return self.out(x)
