from encoder_decoder import subsequent_mask
from data_accessor.data_loader.Settings import *
import torch


def predict(model, inputs):
    input_encoder, input_decoder, embedded_features,encoded_mask = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)
    all_weeks = []
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])
        features = torch.cat([output_prefinal.squeeze(), embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_variance, sales_predictions = model.generate_mu_sigma(features)

        # without teacher forcing
        future_unknown_estimates = sales_predictions

        input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        all_weeks.append(sales_predictions.squeeze())

    return torch.stack(all_weeks).transpose(0, 1)
