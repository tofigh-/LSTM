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

    def forward(self, input, target):
        _assert_no_grad(target)
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
        out = F.mse_loss(input, target, size_average=self.size_average)
        if self.sum_weight:
            out = out / (torch.sum(target) + 0.0001)
        return out


class MDNLOSS(nn.Module):
    def __init__(self):
        super(MDNLOSS, self).__init__()
        self.constant = 1.0 / np.sqrt(2.0 * np.pi)  # normalization factor for Gaussians

    def weighted_logsumexp(self, x, w, dim=None, keepdim=False):
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            # to prevent nasty nan's
            (xm == float('inf')) | (xm == float('-inf')),
            xm,
            xm + torch.log(torch.sum(torch.exp(x - xm) * w, dim, keepdim=True)))
        return x if keepdim else x.squeeze(dim)

    def mdn_loss_stable(self,y, pi, mu, sigma):
        m = torch.distributions.Normal(loc=mu, scale=sigma + 0.1)
        m_lp_y = m.log_prob(y[:, None].expand_as(mu))
        loss = -self.weighted_logsumexp(m_lp_y, pi, dim=1)
        return loss.mean()

    def gaussian_distribution(self, y, mu, sigma):
        # make |mu|=K copies of y, subtract mu, divide by sigma
        result = (y[:, None].expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * self.constant

    def forward(self, pi, sigma, mu, y):
        result = self.mdn_loss_stable(y,pi,mu,sigma)
        if torch.sum(isnan(result)).item():
            import sys
            print "mu"
            print mu
            print "sigma"
            print sigma
            print "after convert to gaussian"
            sys.exit()
        return result


def isnan(x):
    return x != x
