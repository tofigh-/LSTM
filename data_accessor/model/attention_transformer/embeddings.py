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
        batch_size = input.size()[0]
        numeric_features = [input[:, :, self.numeric_feature_indices].float()]
        # Each embedding is computed for the first time-step and replicated 52 times (.expand does exactly that)
        #  for all time steps.
        embedded_input = []
        for i, input_index in enumerate(self.embedding_feature_indices):
            embedded_input.append(
                self.embeddings[i](input[:, 0, input_index].long())
                    .view(-1, 1, self.embedding_sizes[i])
                    .expand(batch_size,self.total_length, self.embedding_sizes[i]))
        past_and_future_sequence = torch.cat(numeric_features + embedded_input, dim=2)
        past_seq = past_and_future_sequence[:, :self.total_input, :]
        future_seq = past_and_future_sequence[:,self.total_input:, :]
        return (past_seq, future_seq)