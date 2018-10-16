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
    input_encoder, input_decoder, embedded_features, encoded_mask = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)
    loss = 0
    all_weeks = []
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])
        features = torch.cat([output_prefinal.squeeze(), embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_predictions = model.generate_mu_sigma(features)

        for country_idx in l2_loss_countries:
            loss += model.loss_weights[country_idx] * loss_function(sales_mean[:, country_idx],
                                                                    sales_future[:, week_idx, country_idx]).sum()
        for country_idx in l1_loss_countries:
            loss += model.loss_weights[country_idx] * loss_function2(sales_mean[:, country_idx],
                                                                     sales_future[:, week_idx, country_idx]).sum()

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
