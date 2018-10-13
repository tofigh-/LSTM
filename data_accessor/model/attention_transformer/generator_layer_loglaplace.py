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
        self.out_sale_beta = nn.Sequential(
            nn.Linear(total_num_features, num_output),
            nn.Softplus()
        )

        self.out_sale_theta = nn.Sequential(
            nn.Linear(total_num_features, num_output),
            nn.Softplus()
        )

    def forward(self, input_values):
        alpha = torch.clamp(self.out_sale_alpha(input_values).squeeze(),
                            min=1e-5,
                            max=1e5)  # (BATCH_SIZE,NUM_OUTPUT)
        beta = torch.clamp(self.out_sale_beta(input_values).squeeze(),
                           min=1e-5,
                           max=1e5)
        theta = self.out_sale_theta(input_values).squeeze()
        if OUTPUT_MODE == 'mode':
            final_output = theta.clone()
            final_output[beta < 1] = 0

        elif OUTPUT_MODE == 'median':
            idx = alpha > beta
            idx_n = 1 - idx
            final_output = theta.clone()
            log_alpha_beta = torch.log(alpha + beta)
            final_output[idx] = final_output[idx] + log_alpha_beta[idx] / beta[idx] - torch.log(2 * alpha[idx]) / beta[
                idx]
            final_output[idx_n] = final_output[idx_n] - log_alpha_beta[idx_n] / alpha[idx_n] + \
                                  torch.log(2 * beta[idx_n]) / alpha[idx_n]

            final_output = torch.clamp(final_output, min=0)
        elif OUTPUT_MODE == 'mean':
            idx = alpha > 1
            final_output = theta.clone()
            final_output[idx] = final_output[idx] + torch.log(alpha[idx] / (alpha[idx] - 1)) + torch.log(
                beta[idx] / (beta[idx] + 1))
            final_output[1 - idx] = 0
            final_output = torch.clamp(final_output, min=0)
        else:
            raise Exception
        return [alpha, beta, theta], final_output
