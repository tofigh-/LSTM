import numpy as np
import torch
from torch.autograd import Variable
from data_accessor.data_loader.Settings import *

use_cuda = torch.cuda.is_available()


def cuda_converter(input):
    if use_cuda:
        return input.cuda()
    else:
        return input


def set_rows(x, x_t, col_index, row_indices):
    temp = x[:, col_index]
    temp.data[row_indices] = x_t[row_indices].data


def set_columns(x, x_t, col_indices, row_index):
    temp = x[row_index, :]
    temp.data[col_indices] = x_t[col_indices].data


def select_rows(x, col_index, row_indices):
    temp = x[:, col_index]
    return temp[row_indices]


def select_cols(x, row_index, col_indices):
    temp = x[row_index, :]
    return temp[col_indices]


def max_likelihood_compute(mu, alpha):
    return torch.clamp(torch.round(mu * (1.0 - alpha)), min=0.0)


def max_likelihood_float_compute(mu, alpha, white_noise=False, noise_std=None):
    output = (mu * (1.0 - alpha))
    if white_noise:
        noise = noise_std * cuda_converter(Variable(torch.randn(output.size())))
        output = output * (1 + noise)
    return torch.clamp(output, min=0.0)


def exponential(input, is_exponential):
    x = np if isinstance(input, np.ndarray) else torch
    if is_exponential:
        return x.exp(input) - 1
    else:
        return input


def log(input, is_log):
    x = np if isinstance(input, np.ndarray) else torch
    if is_log:
        return x.log(input + 1)
    else:
        return input


def threshold(input, th):
    input[input < th] = 0
    return input


def round(input, th):
    input = threshold(input, th)
    return torch.round(input)


def rounder(np_arr):
    return ','.join(['{num:.2f}'.format(num=num) for num in np_arr])

def kpi_compute_per_country(out, target, log_transform, weight=None):
    target_sales = target[SALES_MATRIX][-1, :, :]
    target_global_sales = target[GLOBAL_SALE][-1, :]
    num_sales = target_sales.shape[1]
    kpi_per_country = []
    for i in range(num_sales):
        kpi_per_country.append(kpi_compute(out[:, i], target_sales[:, i], log_transform, weight[:, i]))
    kpi_per_country.append(kpi_compute(
        log(torch.sum(exponential(out, IS_LOG_TRANSFORM), dim=1), IS_LOG_TRANSFORM),
        target_global_sales, log_transform,
        torch.mean(weight,dim=1))
    )
    return np.array(kpi_per_country)


def kpi_compute(out, target, log_transform, weight=None):
    if weight is None:
        return torch.sum(torch.abs(
            exponential(out, log_transform) - exponential(target, log_transform))).item()
    else:
        return torch.sum(weight * torch.abs(
            exponential(out, log_transform) - exponential(target, log_transform))).item()


def complete_embedding_description(embedding_descriptions, label_encoders):
    for feature, description in embedding_descriptions.iteritems():
        if description[EMBEDDING_MAX] is None:
            description[EMBEDDING_MAX] = len(label_encoders[feature])
    return embedding_descriptions
