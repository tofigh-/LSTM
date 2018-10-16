from data_accessor.data_loader.Settings import *
import torch
from data_accessor.model.model_utilities import exponential


def predict_weight_update(model, loss_function, loss_function2, targets_future, inputs, update_weights_mode=False,
                          kpi_loss=None,
                          weights=None):
    sales_future = targets_future[SALES_MATRIX]
    input_encoder, input_decoder, embedded_features, encoded_mask = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)
    all_weeks = []

    loss_l2 = torch.zeros(len(l2_loss_countries))
    loss_l1 = torch.zeros(len(l1_loss_countries))
    week_weights = np.exp(-0.1 * np.arange(OUTPUT_SIZE))
    loss_kpi = 0
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])
        features = torch.cat([output_prefinal.squeeze(), embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_predictions = model.generate_mu_sigma(features)

        # without teacher forcing
        future_unknown_estimates = sales_predictions

        input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        all_weeks.append(sales_mean.squeeze())

        loss_l2 += loss_function(sales_mean[:, l2_loss_countries], sales_future[:, week_idx, l2_loss_countries])
        loss_l1 += loss_function2(sales_mean[:, l1_loss_countries], sales_future[:, week_idx, l1_loss_countries])
        loss_kpi += kpi_loss(exponential(sales_mean, True), exponential(sales_future[:, week_idx, :], True),
                             weights/1000 * week_weights[week_idx])

    return loss_l2, loss_l1, loss_kpi
