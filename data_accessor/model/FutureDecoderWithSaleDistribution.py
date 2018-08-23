import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from time_distributed import TimeDistributed
from data_accessor.data_loader.Settings import *
from my_relu import MyReLU


class FutureDecoderWithSaleDistribution(nn.Module):
    def __init__(self,
                 batch_norm,
                 embedding_descriptions,
                 n_layers=1,
                 hidden_size=None,
                 rnn_layer=None,
                 num_output=1):
        super(FutureDecoderWithSaleDistribution, self).__init__()
        if hidden_size is not None and rnn_layer is not None:
            raise ValueError('If hidden size is provided, lstm layer should be None')
        if hidden_size is None and rnn_layer is None:
            raise ValueError('hidden size and lstm_layer are exclusive parameters. One of them should be set.')
        self.output_size = OUTPUT_SIZE  # 1 in our case where we emit only sales forecast for 1 week at a time.
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.num_output = num_output
        self.embedding_sizes = [embedding_descriptions[feature][EMBEDDING_SIZE] for feature in embedded_features]
        self.embedding_feature_indices = embedding_feature_indices
        self.numeric_feature_indices = numeric_feature_indices
        total_num_features = sum(self.embedding_sizes) + len(self.numeric_feature_indices)

        # It shares the batch_norm layer with encoder
        # (i.e., implicitly assumes encoder and decoder feature inputs have equal dimensions)
        self.batch_norm = batch_norm
        self.relu = nn.Softplus(beta=0.8)
        if rnn_layer is not None:
            self.rnn = rnn_layer
            self.hidden_size = rnn_layer.hidden_size
        else:
            self.rnn = nn.LSTM(input=total_num_features, hidden_size=self.hidden_size, num_layers=n_layers)
        self.factor = 2 if self.rnn.bidirectional else 1
        self.out_sale_fractions = nn.Sequential(
            nn.Linear(self.hidden_size * self.factor, num_output),
            nn.Sigmoid(),
            nn.LogSoftmax(dim=1)
        )
        self.out_global_sale = nn.Sequential(
            nn.Linear(self.hidden_size * self.factor, 1),
            nn.Softplus(beta=0.8)
        )

    def forward(self, input, hidden, embedded_inputs, encoder_outputs=None,
                ):
        # IMPORTANT DECISION: I ASSUME DECODER TAKES THE INPUT IN BATCH BUT TIME STEPS ARE ONE AT A TIME
        # INPUT SIZE: BATCH x TOTAL_FEATURE_NUM
        numeric_features = [input[:, self.numeric_feature_indices].float()]  # BATCH x NUM_NUMERIC_FEATURES
        # Assumption 2: embedded_inputs is a list where each element size: BATCH x EMBEDDING_SIZE
        #  The length of the list is equal to the number of embedded features
        # BATCH_SIZE x TOTAL_NUM_FEAT
        features = self.batch_norm(torch.cat(numeric_features + embedded_inputs, dim=1))
        output, hidden = self.rnn(features.unsqueeze(0), hidden)
        out_global_sale_prediction = self.out_global_sale(output[0]).squeeze()  # (BATCH_SIZE,NUM_OUTPUT)
        out_sale_fractions = self.out_sale_fractions(output[0])
        if len(out_sale_fractions.shape) == 1 and self.num_output > 1:
            out_sale_fractions = out_sale_fractions[None,:]
        return out_global_sale_prediction, out_sale_fractions,hidden
