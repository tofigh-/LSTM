import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from time_distributed import TimeDistributed
from data_accessor.data_loader.Settings import *
from my_relu import MyReLU
from model_utilities import log, exponential


class FutureDecoder(nn.Module):
    def __init__(self,
                 batch_norm,
                 embedding_descriptions,
                 n_layers=1,
                 hidden_size=None,
                 rnn_layer=None,
                 num_output=1):
        super(FutureDecoder, self).__init__()
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
        self.out_sale = nn.Linear(self.hidden_size * self.factor + NUM_COUNTRIES + 1, num_output)

        self.final_out_sale = nn.Sequential(
            nn.Linear(self.hidden_size * self.factor + NUM_COUNTRIES + 1, num_output),
            nn.Tanh()
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
        skiped_inputs = torch.cat(
            [output[0],
             input[:, feature_indices[STOCK]].float(),
             input[:, feature_indices[DISCOUNT_MATRIX]].float()
             ], dim=1)

        out_sales_prediction = self.relu(self.out_sale(skiped_inputs)).squeeze()  # (BATCH_SIZE,NUM_OUTPUT)
        if len(out_sales_prediction.shape) == 1 and self.num_output > 1:
            out_sales_prediction = out_sales_prediction[None, :]
        out_global_sales = log(torch.sum(exponential(out_sales_prediction, IS_LOG_TRANSFORM), dim=1), IS_LOG_TRANSFORM)
        residual_in_normal_domain = self.final_out_sale(skiped_inputs).squeeze()
        # print residual_in_normal_domain
        out_sales_in_normal_domain = self.relu(
            (residual_in_normal_domain.squeeze() + 1) * exponential(out_sales_prediction,
                                                                    IS_LOG_TRANSFORM))
        if len(out_sales_in_normal_domain.shape) == 1 and self.num_output > 1:
            out_sales_in_normal_domain = out_sales_in_normal_domain[None, :]
        global_out_sales_normal_domain = torch.sum(out_sales_in_normal_domain, dim=1)

        return out_global_sales, out_sales_prediction, hidden, global_out_sales_normal_domain, out_sales_in_normal_domain
