from data_accessor.data_loader.Settings import *
import torch
from data_accessor.model.model_utilities import exponential, cuda_converter


def predict_weight_update(model, inputs, targets_future, loss_function=None, loss_function2=None, reference_kpi=None,
                          weights=None,country_id=None):
    sales_future = targets_future[SALES_MATRIX]
    input_encoder, input_decoder, embedded_features, encoded_mask = model.embed(inputs)
    encoder_state = model.encode(input_encoder, encoder_input_mask=encoded_mask)


    loss = 0
    week_weights = np.exp(-0.1 * np.arange(OUTPUT_SIZE))
    for week_idx in range(OUTPUT_SIZE):
        output_prefinal = model.decode(hidden_state=encoder_state, encoder_input_mask=encoded_mask,
                                       decoder_input=input_decoder[:, week_idx:week_idx + 1, :])
        features = torch.cat([output_prefinal[:, 0, :], embedded_features, input_decoder[:, week_idx, :]], dim=1)
        sales_mean, sales_predictions = model.generate_mu_sigma(features)

        # without teacher forcing
        future_unknown_estimates = sales_predictions

        input_decoder[:, week_idx, feature_indices[SALES_MATRIX]] = future_unknown_estimates
        if loss_function is not None:
            loss += loss_function(sales_predictions[:, country_id],
                                  sales_future[:, week_idx, country_id])
        if loss_function2 is not None:
            loss += loss_function2(sales_predictions[:, country_id],
                                   sales_future[:, week_idx, country_id])
        if reference_kpi is not None:
            loss += reference_kpi(exponential(sales_predictions, True), exponential(sales_future[:, week_idx, :], True), weights / 1000 * week_weights[week_idx])

    return loss/OUTPUT_SIZE
