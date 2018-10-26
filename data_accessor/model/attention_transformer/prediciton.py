from data_accessor.data_loader.Settings import *
import torch


def predict(model, loss_function, loss_function2, targets_future, inputs):
    sales_future = targets_future[SALES_MATRIX]
    input_encoder, input_decoder, embedded_features, encoded_mask = model.embed(inputs,mask_key=SALES_MATRIX)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)
    all_weeks = []
    loss = 0
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])
        features = torch.cat([output_prefinal[:,0,:], embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_predictions = model.generate_mu_sigma(features)

        # without teacher forcing
        future_unknown_estimates = sales_predictions

        input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        all_weeks.append(sales_mean.squeeze())
        for country_idx in list_l2_loss_countries:
            loss += model.loss_weights[country_idx] * loss_function(sales_mean[:, country_idx],
                                                                    sales_future[:, week_idx, country_idx]).sum()
        for country_idx in list_l1_loss_countries:
            loss += model.loss_weights[country_idx] * loss_function2(sales_mean[:, country_idx],
                                                                     sales_future[:, week_idx, country_idx]).sum()

    return loss.item() / OUTPUT_SIZE, torch.stack(all_weeks).transpose(0, 1)
