import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _assert_no_grad

from model_utilities import cuda_converter


class LogNegativeBinomial(nn.Module):
    def __init__(self):
        super(LogNegativeBinomial, self).__init__()

    def forward(self, distribution_params, target_values, weights=None):

        if weights is None:
            mu = distribution_params[:, 0]
            alpha = distribution_params[:, 1]
            z = target_values
        else:
            if torch.sum(weights.data) == 0: return 0
            mu = distribution_params[weights, 0]

            alpha = distribution_params[weights, 1]
            z = target_values[weights]
        epsilon = 0.00001
        loss = self.log_gamma(z + 1.0 / (alpha + epsilon)) - self.log_gamma(z + 1.0) - \
               self.log_gamma(1.0 / (alpha + epsilon)) + 1.0 / (alpha + epsilon) * torch.log(
            1.0 / (1.0 + (alpha + epsilon) * mu)) + \
               z * torch.log((alpha + epsilon) * (mu + epsilon) / (1.0 + (alpha + epsilon) * mu))
        # import math
        # if math.isinf(torch.mean(loss).data[0]) or math.isnan(torch.mean(loss).data[0]) :
        #     print (loss.data.cpu().numpy(), z.data.cpu().numpy(),alpha.data.cpu().numpy(),mu.data.cpu().numpy())
        return -torch.mean(loss)

    def _log_gamma_1(self, z):  # Gamma function approximation for z > 0.5
        if z.numel() > 0:
            return (z - 1 / 2.0) * torch.log(z) - z + float((1 / 2.0) * np.log(2 * np.pi)) + 1.0 / (12 * z) - 1.0 / (
                    360 * torch.pow(z, 3)) + 1.0 / (1260 * torch.pow(z, 5))
        else:
            return z

    def _log_gamma_2(self, z):  # Gamma function approximation for z < 0.5
        if z.numel() > 0:
            return float(np.log(np.pi)) - torch.log(torch.sin(np.pi * z)) - self._log_gamma_1(1 - z)
        else:
            return z

    ## This is an approximation of the log Gamma function.
    def log_gamma(self, z):
        out = cuda_converter(Variable(torch.zeros(z.size()[0])))
        out[z <= 0.5] = self._log_gamma_2(z[z <= 0.5])
        out[z > 0.5] = self._log_gamma_1(z[z > 0.5])

        return out


class L1_LOSS(nn.Module):
    def __init__(self, size_average=True, sum_weight=False):
        super(L1_LOSS, self).__init__()
        self.size_average = size_average
        self.sum_weight = sum_weight
        if self.sum_weight: self.size_average = False

    def forward(self, input, target, weights=None):
        _assert_no_grad(target)
        if weights is not None:
            out = torch.mean(torch.mul(F.l1_loss(input, target, size_average=False, reduce=False), weights))
        else:

            out = F.l1_loss(input, target, size_average=self.size_average)
            if self.sum_weight:
                out = out / (torch.sum(target) + 0.0001)
        return out


class L2_LOSS(nn.Module):
    def __init__(self, size_average=True, sum_weight=False):
        super(L2_LOSS, self).__init__()
        self.size_average = size_average
        self.sum_weight = sum_weight
        if self.sum_weight:
            self.size_average = False

    def forward(self, input, target, weights=None):
        _assert_no_grad(target)
        if weights is not None:
            out = torch.mean(torch.mul(F.mse_loss(input, target, size_average=False, reduce=False), weights))
        else:
            out = F.mse_loss(input, target, size_average=self.size_average)
            if self.sum_weight:
                out = out / (torch.sum(target) + 0.0001)
        return out


class LogNormalLoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogNormalLoss, self).__init__()
        self.size_average = size_average

    def forward(self, miu, variance, target):
        _assert_no_grad(target)
        if self.size_average:
            return torch.mean(torch.log(variance) + (target - miu) ** 2 / variance)
        else:
            return torch.sum(torch.log(variance) + (target - miu) ** 2 / variance)
