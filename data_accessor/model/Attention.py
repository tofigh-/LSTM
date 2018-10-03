import torch
import torch.nn as nn
from data_accessor.data_loader.Settings import *
from torch import nn


class TimeDistributedAttention(nn.Module):
    def __init__(self, module, nonlinearity=None):
        super(TimeDistributedAttention, self).__init__()
        self.module = module
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        batch_size,  time_length, num_features = x.shape
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(time_length * batch_size, num_features)
        if self.nonlinearity is not None:
            y = self.nonlinearity(self.module(x_reshape))
        else:
            y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(batch_size, time_length, y.shape[1])
        return y


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         attention = Attention(256)
         query = torch.randn(5, 1, 256)
         context = torch.randn(5, 5, 256)
         output, weights = attention(query, context)
         output.size()
         torch.Size([5, 1, 256])
         weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, dropout=ATTENTION_DROPOUT, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Sequential(nn.Linear(dimensions, dimensions, bias=False),
                                           nn.Dropout(dropout))
        self.attention = TimeDistributedAttention(nn.Linear(dimensions * 2,1),nn.Tanh())
        self.linear_out = nn.Sequential(
            nn.Linear(dimensions * 2, dimensions, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, context, mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query_expanded = query.view(batch_size, output_len, dimensions).expand(-1, query_len, -1)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = self.attention(torch.cat([query_expanded,context],dim=2))
        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix.squeeze(), query), dim=1)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)

        return output, attention_weights
