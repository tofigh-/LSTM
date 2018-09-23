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
        total_num_features = sum(self.embedding_sizes) + len(self.numeric_feature_indices)

        # It turns out I'm not normalizing, I'm transforming. That was a surprise. What happened to BatchNorm1d!?
        self.batch_norm = nn.Sequential(
            nn.Linear(in_features=total_num_features, out_features=total_num_features))
        self.time_dist_batch_norm = TimeDistributed(self.batch_norm)

        self.p = rnn_dropout
        self.rnn = nn.LSTM(input_size=total_num_features, hidden_size=hidden_size, num_layers=n_layers,
                           bidirectional=bidirectional)
        self.hidden_state_dimensionality_reduction = nn.Sequential(
            nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            nn.Softplus()
        )

        self.first_week_mean_sale = nn.Sequential(
            # nn.BatchNorm1d(hidden_size),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, 14),
            nn.Softplus()
        )

        self.first_week_variance_sale = nn.Sequential(
            # self.first_week_mean_sale._modules['0'],
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, 14),
            nn.Softplus()
        )

    def forward(self, input, hidden):
        batch_size = input.size()[1]
        numeric_features = [input[:, :, self.numeric_feature_indices].float()]
        # Each embedding is computed for the first time-step and replicated 52 times (.expand does exactly that)
        #  for all time steps.
        embedded_input = []
        for i, input_index in enumerate(self.embedding_feature_indices):
            embedded_input.append(
                self.embeddings[i](input[0, :, input_index].long())
                    .view(1, -1, self.embedding_sizes[i])
                    .expand(PAST_KNOWN_LENGTH, batch_size, self.embedding_sizes[i]))

        # concat on the last axis which is the feature axis:
        #  With this we have all the dynamic and static features in one tensor
        output = F.dropout(self.time_dist_batch_norm(torch.cat(numeric_features + embedded_input, dim=2).contiguous()),
                           self.p)
        output, hidden = self.rnn(output, hidden)
        hidden_out = (
            self.hidden_state_dimensionality_reduction(torch.cat([hidden[0][0], hidden[0][1]], dim=1))[None, :, :],
            self.hidden_state_dimensionality_reduction(torch.cat([hidden[1][0], hidden[1][1]], dim=1))[None, :, :]
        )
        out_first_week_mean_prediction = self.first_week_mean_sale(hidden_out[0]).squeeze()
        out_first_week_variance_prediction = torch.clamp(
            self.first_week_variance_sale(hidden_out[0]).squeeze(),
            min=1e-5,
            max=1e5
        )
        out_first_week_prediction = out_first_week_mean_prediction + 0.5 * out_first_week_variance_prediction
        return output, \
               hidden_out, \
               [embedded_feature[0, :, :] for embedded_feature in embedded_input], \
               out_first_week_prediction, \
               out_first_week_mean_prediction, \
               out_first_week_variance_prediction

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
