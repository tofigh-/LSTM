import torch
from torch import nn


class GeneratorLayer(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, total_num_features, num_output):
        super(GeneratorLayer, self).__init__()
        self.out_sale_means = nn.Sequential(
            nn.Linear(total_num_features, num_output),
            nn.Softplus()
        )

    def forward(self, input_values):
        out_sales_mean_predictions = self.out_sale_means(input_values).squeeze()  # (BATCH_SIZE,NUM_OUTPUT)

        final_output = out_sales_mean_predictions
        return out_sales_mean_predictions, final_output
