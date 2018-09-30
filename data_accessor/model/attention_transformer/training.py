from encoder_decoder import subsequent_mask
from data_accessor.data_loader.Settings import *

import random
import math
import torch
import sys


def train_per_batch(model, inputs, targets_future, loss_function, loss_function2,
                    teacher_forcing_ratio):
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    sales_future = targets_future[SALES_MATRIX]
    input_encoder, input_decoder = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=None)
    loss = 0
    all_weeks = []
    for week_idx in range(input_decoder.shape[1]):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=None,
                                       decoder_input=input_decoder[:, 0:week_idx + 1, :],
                                       input_decoder_mask=
                                       subsequent_mask(week_idx + 1))
        sales_mean, sales_variance, sales_predictions = model.generate_mu_sigma(output_prefinal)
        if len(sales_predictions.shape) == 2:
            sales_predictions = sales_predictions.unsqueeze(1)
            sales_mean = sales_mean.unsqueeze(-1)
            sales_variance = sales_variance.unsqueeze(-1)
        loss += loss_function(sales_mean[:, -1, :], sales_variance[:, -1, :], sales_future[:, week_idx, :])

        if use_teacher_forcing:
            input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = sales_future[:, week_idx, :].data

        else:
            # without teacher forcing
            future_unknown_estimates = sales_predictions.detach()
            # Batch x time x num_feature
            input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates[:, -1, :]
        all_weeks.append(sales_predictions.squeeze())

    sales_predictions = torch.stack(all_weeks).transpose(1, 0)
    if math.isnan(loss.item()):
        print "loss is ", loss
        print "sum input 0 ", torch.sum(inputs[0])
        print "sum input 1 ", torch.sum(inputs[1])
        sys.exit()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item(), sales_predictions
