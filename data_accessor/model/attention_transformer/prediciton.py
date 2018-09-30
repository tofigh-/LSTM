from encoder_decoder import subsequent_mask
from data_accessor.data_loader.Settings import *
import torch


def predict(model, inputs):
    input_encoder, input_decoder = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=None)
    all_weeks = []
    for week_idx in range(input_decoder.shape[1]):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=None,
                                       decoder_input=input_decoder[:, 0: week_idx + 1, :],
                                       input_decoder_mask=subsequent_mask(week_idx + 1))
        sales_mean, sales_variance, sales_predictions = model.generate_mu_sigma(output_prefinal)

        # without teacher forcing
        future_unknown_estimates = sales_predictions.detach()

        input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        all_weeks.append(sales_predictions.squeeze())

    return torch.stack(all_weeks).transpose(1, 0)
