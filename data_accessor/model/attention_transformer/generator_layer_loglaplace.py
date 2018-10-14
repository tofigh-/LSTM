import torch
from torch import nn
from data_accessor.data_loader.Settings import OUTPUT_MODE


class GeneratorLayerLogLaplace(nn.Module):

    def __init__(self, total_num_features, num_output):
        super(GeneratorLayerLogLaplace, self).__init__()
        self.out_sale_alpha = nn.Sequential(
            nn.Linear(total_num_features, num_output),
            nn.Softplus()
        )

        self.out_sale_theta = nn.Sequential(
            nn.Linear(total_num_features, num_output),
            nn.Softplus()
        )

    def forward(self, input_values):
        alpha = torch.clamp(self.out_sale_alpha(input_values).squeeze(),
                            min=0.5,
                            max=1.5)  # (BATCH_SIZE,NUM_OUTPUT)
        theta = self.out_sale_theta(input_values).squeeze()
        if OUTPUT_MODE == 'mode':
            final_output = theta.clone()
            final_output[alpha < 1] = 0

        elif OUTPUT_MODE == 'median':
            final_output = theta.clone()

        elif OUTPUT_MODE == 'mean':
            idx = alpha > 1
            final_output = theta.clone()
            final_output[idx] = final_output[idx] + torch.log(1 + 1 / (alpha[idx] ** 2 - 1))
            final_output[1-idx] = 0
        else:
            raise Exception
        return [alpha, theta], final_output
