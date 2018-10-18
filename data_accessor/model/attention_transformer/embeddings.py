import torch.nn as nn
from data_accessor.data_loader.Settings import *
import torch


class Embeddings(nn.Module):
    def __init__(self, embedding_descriptions, total_input, forecast_length):
        super(Embeddings, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(
                num_embeddings=embedding_descriptions[feature][EMBEDDING_MAX],
                embedding_dim=embedding_descriptions[feature][EMBEDDING_SIZE]
            )

                for feature in embedded_features
            ]
        )

        self.embedding_sizes = [embedding_descriptions[feature][EMBEDDING_SIZE] for feature in embedded_features]
        self.embedding_feature_indices = embedding_feature_indices
        self.numeric_feature_indices = numeric_feature_indices
        self.total_length = total_input + forecast_length
        self.total_input = total_input

    def forward(self, input):
        numeric_features = input[:, :, self.numeric_feature_indices].float()
        numeric_mask = input[:, :, feature_indices[STOCK]] > 0
        # Each embedding is computed for the first time-step and replicated 52 times (.expand does exactly that)
        #  for all time steps.
        embedded_input = []
        for i, input_index in enumerate(self.embedding_feature_indices):
            embedded_input.append(
                self.embeddings[i](input[:, 0, input_index].long()).unsqueeze(-1)
            )
        past_seq = numeric_features[:, :self.total_input, :]
        future_seq = numeric_features[:, self.total_input:, :]
        embedded_output = torch.sum(torch.cat(embedded_input, dim=2), dim=2)
        return past_seq, future_seq, embedded_output, numeric_mask[:, :self.total_input, :].transpose(1, 2)
