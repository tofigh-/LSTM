from encoder_decoder import subsequent_mask
from data_accessor.data_loader.Settings import *

import random
import math
import torch
import sys
import numpy as np


def train_per_batch(model, inputs, targets_future, loss_function, loss_function2, bias_loss, loss_masks,
                    teacher_forcing_ratio):
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    sales_future = targets_future[SALES_MATRIX]
    mask_key = STOCK
    input_encoder, input_decoder, embedded_features, encoded_mask = model.embed(inputs, mask_key)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)
    loss = 0
    all_weeks = []
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])

        features = torch.cat([output_prefinal[:, 0, :], embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_predictions = model.generate_mu_sigma(features)

        l2 = loss_function(sales_predictions[loss_masks[:, week_idx]],
                           sales_future[loss_masks[:, week_idx], week_idx, :])
        l1 = loss_function2(sales_predictions[loss_masks[:, week_idx]],
                            sales_future[loss_masks[:, week_idx], week_idx, :])
        loss += (torch.cat([l2, l1]) * model.loss_weights).sum()
        if bias_loss is not None:
            loss += bias_loss(sales_predictions[loss_masks[:, week_idx]],
                              sales_future[loss_masks[:, week_idx], week_idx, :])

        if use_teacher_forcing:
            input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = sales_future[:, week_idx, :].data

        else:
            # without teacher forcing
            future_unknown_estimates = sales_predictions
            # Batch x time x num_feature
            input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        all_weeks.append(sales_predictions.squeeze())

    sales_predictions = torch.stack(all_weeks).transpose(0, 1)
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
