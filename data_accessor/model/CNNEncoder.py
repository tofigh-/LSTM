"""
LICENSE: https://github.com/allenai/allennlp/blob/master/LICENSE
"""

import torch
from torch.nn import Conv1d, Linear, ReLU


class CNNEncoder(torch.nn.Module):

    def __init__(self,
                 embedding_dim,
                 num_filters,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=ReLU(),
                 output_dim=None):
        super(CNNEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation
        self._output_dim = output_dim

        self._convolution_layers = [
            Conv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self):
        return self._embedding_dim


    def get_output_dim(self):
        return self._output_dim


    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encode.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(
            filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result