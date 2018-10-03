import torch
import torch.nn as nn

from data_accessor.data_loader.Settings import *
from model_utilities import cuda_converter
from time_distributed import TimeDistributed
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size,
                 embedding_descriptions,
                 n_layers=1,
                 rnn_dropout=0.1,
                 bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embeddings = nn.ModuleList(
            [
                cuda_converter(
                    nn.Embedding(
                        num_embeddings=embedding_descriptions[feature][EMBEDDING_MAX],
                        embedding_dim=embedding_descriptions[feature][EMBEDDING_SIZE]
                    )
                )
                for feature in embedded_features
            ]
        )

        self.embedding_sizes = [embedding_descriptions[feature][EMBEDDING_SIZE] for feature in embedded_features]
        self.embedding_feature_indices = embedding_feature_indices
        self.numeric_feature_indices = numeric_feature_indices
        self.p = rnn_dropout
        self.rnn = nn.LSTM(input_size=len(self.numeric_feature_indices), hidden_size=hidden_size, num_layers=n_layers,
                           bidirectional=bidirectional)
        self.hidden_state_dimensionality_reduction = nn.Sequential(
            nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            nn.Softplus()
        )

    def forward(self, input, hidden):
        numeric_features = input[:, :, self.numeric_feature_indices].float()
        encoder_mask = input[:, :, feature_indices[STOCK]] > 0
        embedded_input = []
        for i, input_index in enumerate(self.embedding_feature_indices):
            embedded_input.append(
                self.embeddings[i](input[0, :, input_index].long()))

        output, hidden = self.rnn(numeric_features, hidden)
        hidden_out = (
            self.hidden_state_dimensionality_reduction(torch.cat([hidden[0][0], hidden[0][1]], dim=1))[None, :, :],
            self.hidden_state_dimensionality_reduction(torch.cat([hidden[1][0], hidden[1][1]], dim=1))[None, :, :]
        )
        if self.bidirectional:
            output = torch.cat([output[:, :, 0:self.hidden_size], output[:, :, self.hidden_size:]], dim=0).transpose(0,
                                                                                                                     1)
            encoder_mask = encoder_mask.repeat(2, 1, 1).transpose(0, 1).squeeze()
        else:
            output = output.transpose(0, 1)
            encoder_mask = encoder_mask.transpose(0,1).squeeze()
        return output, \
               hidden_out, \
               torch.cat(embedded_input, dim=1), \
               encoder_mask

    def initHidden(self, batch_size):
        factor = 2 if self.bidirectional else 1
        if self.rnn.mode == 'LSTM':
            result = (
                cuda_converter(torch.zeros(self.n_layers * factor, batch_size, self.hidden_size)),
                cuda_converter(torch.zeros(self.n_layers * factor, batch_size, self.hidden_size))
            )
        else:
            result = cuda_converter(torch.zeros(self.n_layers * factor, batch_size, self.hidden_size))
        return result
