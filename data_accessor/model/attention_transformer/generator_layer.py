import torch
from torch import nn


class GeneratorLayer(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, num_output):
        super(GeneratorLayer, self).__init__()
        self.out_sale_means = nn.Sequential(
            nn.Linear(d_model, num_output),
            nn.Softplus()
        )
        self.out_sale_variances = nn.Sequential(
            nn.Linear(d_model, num_output),
            nn.Softplus()
        )

    def forward(self, input_values):
        out_sales_mean_predictions = self.out_sale_means(input_values).squeeze()  # (BATCH_SIZE,NUM_OUTPUT)
        out_sales_variance_predictions = torch.clamp(self.out_sale_variances(input_values).squeeze(),
                                                     min=1e-5,
                                                     max=1e5)
        final_output = out_sales_mean_predictions + 0.5 * out_sales_variance_predictions
        return out_sales_mean_predictions, out_sales_variance_predictions, final_output
