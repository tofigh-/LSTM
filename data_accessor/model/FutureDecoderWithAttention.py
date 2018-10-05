import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from time_distributed import TimeDistributed
from data_accessor.data_loader.Settings import *
from my_relu import MyReLU
from model_utilities import log, exponential
from Attention import Attention


class FutureDecoderWithAttention(nn.Module):
    def __init__(self, embedding_descriptions,
                 n_layers=1,
                 hidden_size=None,
                 rnn_layer=None,
                 num_output=1):
        super(FutureDecoderWithAttention, self).__init__()
        self.output_size = OUTPUT_SIZE  # 1 in our case where we emit only sales forecast for 1 week at a time.
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.num_output = num_output
        self.embedding_sizes = [embedding_descriptions[feature][EMBEDDING_SIZE] for feature in embedded_features]
        self.embedding_feature_indices = embedding_feature_indices
        self.numeric_feature_indices = numeric_feature_indices
        total_num_features = sum(self.embedding_sizes) + len(self.numeric_feature_indices)
        self.num_output = num_output
        # It shares the batch_norm layer with encoder
        # (i.e., implicitly assumes encoder and decoder feature inputs have equal dimensions)
        self.relu = nn.Softplus(beta=0.8)
        self.rnn = nn.LSTM(input_size=len(self.numeric_feature_indices), hidden_size=self.hidden_size,
                           num_layers=n_layers)
        self.attention = Attention(dimensions=hidden_size)
        if rnn_layer is not None:
            self.rnn.weight_ih_l0 = rnn_layer.weight_ih_l0
            self.rnn.weight_hh_l0 = rnn_layer.weight_hh_l0
            self.rnn.bias_ih_l0 = rnn_layer.bias_ih_l0
            self.rnn.bias_hh_l0 = rnn_layer.bias_hh_l0

        self.out_sale_means = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hidden_size + total_num_features, 1), nn.Softplus()) for _ in
             range(num_output)]
        )

        self.out_sale_variances = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.hidden_size + total_num_features, 1), nn.Softplus()) for _ in
             range(num_output)]
        )

    def forward(self, input, hidden, embedded_inputs, encoder_outputs=None, encoder_mask=None):
        # INPUT SIZE: BATCH x TOTAL_FEATURE_NUM
        numeric_features = input[:, self.numeric_feature_indices].float()  # BATCH x NUM_NUMERIC_FEATURES
        features = torch.cat([numeric_features, embedded_inputs], dim=1)
        output, hidden = self.rnn(numeric_features.unsqueeze(0), hidden)
        attended_features, _ = self.attention(query=output.transpose(0, 1), context=encoder_outputs, mask=encoder_mask)
        encoded_features = torch.cat([attended_features.squeeze(), features], dim=1)
        out_sales_mean_predictions = torch.cat(
            [self.out_sale_means[i](encoded_features) for i in range(self.num_output)]
            , dim=1)  # (BATCH_SIZE,NUM_OUTPUT)
        out_sales_variance_predictions = torch.cat(
            [
                torch.clamp(self.out_sale_variances[i](encoded_features),
                            min=1e-5,
                            max=1e5) for i in range(self.num_output)
            ]
            , dim=1)
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
