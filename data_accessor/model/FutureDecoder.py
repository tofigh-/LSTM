import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from time_distributed import TimeDistributed
from data_accessor.data_loader.Settings import *
from my_relu import MyReLU
from model_utilities import log, exponential

from WeightDrop import WeightDrop


class FutureDecoder(nn.Module):
    def __init__(self, embedding_descriptions,
                 n_layers=1,
                 hidden_size=None,
                 rnn_layer=None,
                 num_output=1):
        super(FutureDecoder, self).__init__()
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
        self.relu = nn.Softplus(beta=0.8)
        self.lstm = nn.LSTM(input_size=len(self.numeric_feature_indices), hidden_size=self.hidden_size,
                            num_layers=n_layers)
        self.rnn_layer = rnn_layer
        if rnn_layer is not None:
            self.lstm.weight_ih_l0 = rnn_layer.module.weight_ih_l0
            del self.lstm._parameters['weight_hh_l0']
            self.lstm.register_parameter('weight_hh_l0_raw', self.rnn_layer.module.weight_hh_l0_raw)
            self.lstm.bias_ih_l0 = rnn_layer.module.bias_ih_l0
            self.lstm.bias_hh_l0 = rnn_layer.module.bias_hh_l0

        # self.rnn = WeightDrop(self.lstm, weights=['weight_hh_l0'], dropout=RNN_DROPOUT)
        self.rnn = self.lstm
        self.out_sale_means = nn.Sequential(
            nn.Linear(self.hidden_size + total_num_features, num_output),
            nn.Softplus()
        )
        self.out_sale_variances = nn.Sequential(
            nn.Linear(self.hidden_size + total_num_features, num_output),
            nn.Softplus()
        )

    def forward(self, input, hidden, embedded_inputs, encoder_outputs=None,
                ):
        # IMPORTANT DECISION: I ASSUME DECODER TAKES THE INPUT IN BATCH BUT TIME STEPS ARE ONE AT A TIME
        # INPUT SIZE: BATCH x TOTAL_FEATURE_NUM
        numeric_features = input[:, self.numeric_feature_indices].float()  # BATCH x NUM_NUMERIC_FEATURES
        # Assumption 2: embedded_inputs is a list where each element size: BATCH x EMBEDDING_SIZE
        #  The length of the list is equal to the number of embedded features
        # BATCH_SIZE x TOTAL_NUM_FEAT
        features = torch.cat([numeric_features, embedded_inputs], dim=1)
        setattr(self.rnn, 'weight_hh_l0', self.rnn_layer.module.weight_hh_l0)
        output, hidden = self.rnn(numeric_features.unsqueeze(0), hidden)
        encoded_features = torch.cat([output[0], features], dim=1)
        out_sales_mean_predictions = self.out_sale_means(encoded_features).squeeze()  # (BATCH_SIZE,NUM_OUTPUT)
        out_sales_variance_predictions = torch.clamp(self.out_sale_variances(encoded_features).squeeze(),
                                                     min=1e-5,
                                                     max=1e5)  # (BATCH_SIZE,NUM_OUTPUT)
        if len(out_sales_mean_predictions.shape) == 1 and self.num_output > 1:
            out_sales_mean_predictions = out_sales_mean_predictions[None, :]
            out_sales_variance_predictions = out_sales_variance_predictions[None, :]

        out_global_sales = log(
            torch.sum(exponential(out_sales_mean_predictions + 0.5 * out_sales_variance_predictions, IS_LOG_TRANSFORM),
                      dim=1), IS_LOG_TRANSFORM)
        return out_global_sales, \
               (out_sales_mean_predictions + 0.5 * out_sales_variance_predictions), \
               out_sales_mean_predictions, \
               out_sales_variance_predictions, \
               hidden


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)
