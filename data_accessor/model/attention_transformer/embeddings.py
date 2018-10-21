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
        self.num_numeric_features = len(self.numeric_feature_indices)

    def forward(self, input):
        numeric_features = input[:, :, self.numeric_feature_indices].float()
        if sort_key == SALES_MATRIX:
            numeric_mask = (
                    (torch.sum(input[:, :self.total_input, feature_indices[SALES_MATRIX]], dim=2)) > 0).unsqueeze(
                -1)
        if sort_key == STOCK:
            numeric_mask = (input[:, :self.total_input, feature_indices[STOCK]]) > 0
        sum_mask = torch.sum(numeric_mask, dim=1)
        batch_size = sum_mask.shape[0]
        sum_mask_fist_element = sum_mask[0, 0].item()
        all_seq_has_equal_length = (sum_mask == sum_mask_fist_element).sum() == batch_size

        # Each embedding is computed for the first time-step and replicated 52 times (.expand does exactly that)
        #  for all time steps.
        embedded_input = []
        for i, input_index in enumerate(self.embedding_feature_indices):
            embedded_input.append(
                self.embeddings[i](input[:, 0, input_index].long()).unsqueeze(-1)
            )
        past_seq = numeric_features[:, :self.total_input, :]
        if all_seq_has_equal_length:
            if sum_mask_fist_element != TOTAL_INPUT and sum_mask_fist_element != 0:
                past_seq = torch.masked_select(past_seq, numeric_mask).view(batch_size, -1,
                                                                            self.num_numeric_features)
            if sum_mask_fist_element ==0:
                past_seq = past_seq[:,[-1],:]
            numeric_mask = None

        future_seq = numeric_features[:, self.total_input:, :]
        embedded_output = torch.sum(torch.cat(embedded_input, dim=2), dim=2)
        if numeric_mask is not None:
            numeric_mask = numeric_mask.transpose(1, 2)
        print past_seq.shape[1]
        return past_seq, future_seq, embedded_output, numeric_mask
