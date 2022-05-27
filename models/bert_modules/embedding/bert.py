import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, user_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=embed_size)
        self.user_embedding = TokenEmbedding(
            vocab_size=user_size, embed_size=embed_size)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, inputs):
        # + self.segment(segment_label)
        user_id, product_history, target_product_id,  product_history_ratings = inputs

        x = self.token(product_history) + self.position(product_history)
        B, T = product_history_ratings.shape
        user_info = self.user_embedding(user_id).view(B, 1, -1)
        target_info = self.token(target_product_id).view(B, 1, -1)
        x = x*product_history_ratings.view(B, T, 1)
        x = torch.cat([user_info, x, target_info], dim=1)
        return self.dropout(x)
