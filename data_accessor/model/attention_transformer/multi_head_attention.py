import torch
import math
from torch.nn import functional as F
import torch.nn as nn
from encoder import clones
from attention_time_distributed import TimeDistributed


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.similarity = nn.Linear(in_features=2 * self.d_k, out_features=1)
        self.distributed_similarity = TimeDistributed(self.similarity, nn.Tanh())
        self.num_attention_head = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, self_attention=True):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.shape[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [layer(x).view(batch_size, -1, self.num_attention_head, self.d_k).transpose(1, 2) for layer, x in
             zip(self.linears[0:3], (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        value_weighted_sum, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout,
                                                       self_attention=self_attention)

        # 3) "Concat" using a view and apply a final linear.
        value_weighted_sum = value_weighted_sum.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_attention_head * self.d_k)
        out = self.linears[-1](value_weighted_sum)
        return out

    def attention(self, query, key, value, mask=None, dropout=None, self_attention=True):
        "Compute 'Scaled Dot Product Attention'"
        batch_size, num_head, query_time_length, d_k = query.shape
        if self_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                     / math.sqrt(d_k)
        else:
            n_batch, num_head, time_length, num_features = key.shape
            scores = self.distributed_similarity(
                torch.cat([query.expand(n_batch, num_head, time_length, num_features), key], dim=3))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            if query_time_length > 1:
                scores = scores.masked_fill(mask.transpose(-2, -1) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value)
        return output, p_attn
