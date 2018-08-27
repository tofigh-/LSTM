import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from torch.distributions.gumbel import Gumbel
import sys
from model_utilities import cuda_converter

class MDN(nn.Module):
    def __init__(self, input, n_gaussians):
        super(MDN, self).__init__()
        self.z_pi = nn.Linear(input, n_gaussians)
        self.z_sigma = nn.Linear(input, n_gaussians)
        self.z_mu = nn.Linear(input, n_gaussians)
        self.mu_old = None
        self.sigma_old = None
        self.pi_old = None
    def forward(self, x):
        z_h = F.tanh(x)
        pi = F.softmax(F.sigmoid(self.z_pi(z_h)).clamp(min=0.00001), -1)
        sigma = (torch.exp(self.z_sigma(z_h))).clamp(min=1)
        mu = (torch.exp(self.z_mu(z_h))).clamp(min=0.00001)
        if torch.sum(torch.isnan(mu)).item() or float('inf') == torch.mean(mu).item():
            print "Mu is nan for unknown reason"
            print mu
            print "%%%%%%%%%%%%%%"
            print self.mu_old
            print self.sigma_old
            print self.pi_old
            sys.exit()
        else:
            self.mu_old = mu
            self.sigma_old = sigma
            self.pi_old = pi

        if torch.sum(torch.isnan(sigma)).item() or float('inf') == torch.mean(sigma).item():
            print "sigma is nan for unknown reason"
            print sigma
            print "%%%%%%%%%%%%%%"
            print self.mu_old
            print self.sigma_old
            print self.pi_old
            sys.exit()

        if torch.sum(torch.isnan(pi)).item() or float('inf') == torch.mean(pi).item():
            print "pi is nan for unknown reason"
            print pi
            print "%%%%%%%%%%%%%%"
            print self.mu_old
            print self.sigma_old
            print self.pi_old
            sys.exit()

        return pi, mu, sigma

    @staticmethod
    def gumbel_class_sample(p_matrix, axis=1, soft=False):
        # Select one out o K pi values for each row of p_matrix
        z = cuda_converter(Gumbel(loc=0, scale=1).sample(sample_shape=p_matrix.shape))
        noisy_samples = (torch.log(p_matrix) + z)
        if soft:
            out = F.softmax(noisy_samples, dim=axis)
        else:
            out = noisy_samples.argmax(dim=axis)
        return out

    @staticmethod
    def deterministic_class_sample(p_matrix, axis=1):
        return p_matrix.argmax(axis=axis)

    @staticmethod
    def probabilistic_inference(num_samples, p_matrix, sigma_data, mu_data, soft=False):
        selected_classes = MDN.gumbel_class_sample(p_matrix, axis=1, soft=soft)
        if soft:
            rn = cuda_converter(torch.randn(sigma_data.shape))
            sampled_data = F.relu(torch.sum(selected_classes * (mu_data + rn * sigma_data), dim=1))

        else:
            indices = (cuda_converter(torch.arange(0, num_samples, out=torch.LongTensor())), selected_classes)
            rn = cuda_converter(torch.randn(num_samples))
            sampled_data = F.relu(rn * sigma_data[indices] + mu_data[indices])
        return sampled_data

    @staticmethod
    def deterministic_inference(num_samples, p_matrix, mu_data):
        selected_classes = MDN.deterministic_class_sample(p_matrix)
        indices = (np.arange(num_samples), selected_classes)
        sampled_data = mu_data[indices]
        return sampled_data

    @staticmethod
    def expected_inference(p_matrix, mu_data):
        expected_output = torch.sum(p_matrix * mu_data, dim=1)
        return expected_output
