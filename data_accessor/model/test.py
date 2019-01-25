from os.path import join
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_accessor.data_loader.Settings import *
from data_accessor.model.attention_transformer.predict_per_batch import predict_per_batch
from data_accessor.model.loss import L2Loss, L1Loss, KPILoss
from model_utilities import exponential, cuda_converter, \
    kpi_compute_per_country, rounder

import sys

from data_accessor.model.attention_transformer.train_per_batch import train_per_batch
from data_accessor.model.attention_transformer.encoder_decoder import EncoderDecoder
from datetime import datetime


class Testing(object):
    def __init__(self, model, train_dataloader, test_dataloader, n_iters, output_size,
                 total_input, label_encoder):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.output_size = output_size
        self.total_input = total_input
        self.total_length = self.output_size + self.total_input
        self.n_iters = n_iters
        self.msloss = L2Loss(sum_loss=SUM_LOSS)
        self.l1loss = L1Loss(sum_loss=SUM_LOSS)
        self.kpi_loss = KPILoss()
        self.train_far_futures = False
        self.bias_loss = None
        self.cache_validation = None
        self.cg2_encoder = {int(v): k for k, v in label_encoder[CG2].iteritems()}

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.total_length = self.output_size + self.total_input
        self.train_dataloader.set_forecast_length(self.output_size)
        self.test_dataloader.set_forecast_length(self.output_size)

    def _kpi_print(self, mode, loss_value, kpi_value, weekly_aggregated_kpi, weekly_aggregated_kpi_scale, k1, k2,
                   predicted_country_sales=None, country_sales=None):
        print "{mode} Loss: {loss}".format(mode=mode, loss=loss_value)
        print "National {mode} Sale KPI {kpi}".format(mode=mode, kpi=kpi_value)

        print "Weekly {mode} Aggregated KPI {kpi}".format(mode=mode,
                                                          kpi=rounder(np.sum(weekly_aggregated_kpi, axis=0) / np.sum(
                                                              weekly_aggregated_kpi_scale, axis=0) * 100)
                                                          )
        global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(self.output_size)]
        print "National AVG {mode} KPI is {t_kpi}".format(mode=mode, t_kpi=global_kpi)
        if predicted_country_sales is not None:
            bias = [predicted_country_sales[i] / country_sales[i] for i in range(self.output_size)]
            print "Bias {mode} per country per week {bias}".format(mode=mode, bias=bias)

    def _mini_batch_preparation(self, batch_data, iter_num):
        targets_future = dict()

        #  BATCH x OUTPUT x NUM_COUNTRY
        max_stock = np.max(batch_data[:, :, feature_indices[STOCK][0]], axis=1)
        targets_future[SALES_MATRIX] = cuda_converter(torch.from_numpy(
            batch_data[:, self.total_input:, feature_indices[SALES_MATRIX]].copy()
        ).float())

        targets_future[GLOBAL_SALE] = cuda_converter(torch.from_numpy(
            batch_data[:, self.total_input:, feature_indices[GLOBAL_SALE][0]].copy()
        ).float())

        targets_future[STOCK] = cuda_converter(torch.from_numpy(
            batch_data[:, self.total_input:, feature_indices[STOCK][0]].copy()
        ).float())
        batch_data[:, self.total_input:, feature_indices[SALES_MATRIX]] = batch_data[:,
                                                                          self.total_input - 1:self.total_input,
                                                                          feature_indices[SALES_MATRIX]]
        batch_data[:, self.total_input:, feature_indices[GLOBAL_SALE][0]] = batch_data[:,
                                                                            self.total_input - 1:self.total_input,
                                                                            feature_indices[GLOBAL_SALE][0]]
        batch_data[:, self.total_input:, feature_indices[STOCK][0]] = np.log1p(np.expm1(max_stock) * iter_num / 10.0)[:,
                                                                      None]
        black_price = exponential(
            cuda_converter(
                torch.from_numpy(
                    batch_data[:, self.total_input - 1:self.total_input,
                    feature_indices[BLACK_PRICE_INT]]).float().squeeze()
            ),
            IS_LOG_TRANSFORM)
        return targets_future, batch_data, black_price

    def _data_iter(self, iter_num, data, loss_func, loss_func2, model_mode="train_near_future"):
        self.model.mode(mode=model_mode)
        total_predictions = []
        for batch_num, batch_data2 in enumerate(data):
            loss_masks = []
            batch_data = []
            batch_data[:], loss_masks[:] = zip(*batch_data2)
            batch_data = np.array(batch_data)
            # Batch x time x num
            if batch_data.shape[0] == 1:
                print "Warning; batch size is one"
                continue
            x, y, z = np.where(np.isinf(batch_data))
            if len(z) > 0:
                print "these feature indices are inf: ", z
                print feature_indices
                sys.exit()

            targets_future, batch_data, black_price = self._mini_batch_preparation(batch_data, iter_num)

            loss, sale_predictions = predict_per_batch(
                model=self.model,
                loss_function=loss_func,
                loss_function2=loss_func2,
                bias_loss=self.bias_loss,
                targets_future=targets_future,
                inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous(),
                output_size=self.output_size
            )
            unbounded_prediction = np.sum(exponential(sale_predictions, True).squeeze().data.cpu().numpy(), axis=1)
            out_prediction = np.minimum(unbounded_prediction, batch_data[:, self.total_input, feature_indices[STOCK][0]])
            cg2 = batch_data[:, 0, feature_indices[CG2]]
            cg2 = np.array([self.cg2_encoder[int(el)] for el in cg2])[:, None]
            total_predictions.append(np.concatenate([out_prediction[:, None], cg2], axis=1))

        return np.concatenate(total_predictions)

    def test(self):
        EncoderDecoder.load_checkpoint({ENCODER_DECODER_CHECKPOINT: 'transformer_sibr_model_s3.torch'}, self.model)

        self.model.mode(mode=PREDICT)
        out = self._test()
        out = np.concatenate([out[:, 0:-1:2], out[:, -1][:, None]], axis=1)
        header= map(lambda i: str(i), range(out.shape[1] -1)) + ["cg2"]
        out_df = pd.DataFrame(out,columns=header)
        out_df.to_csv("stock_Response_withcg2.csv",index=False)

    def _test(self):
        loss_function = self.msloss
        loss_function2 = self.l1loss
        self.model.mode(mode=PREDICT)
        row_iteration_order = self.test_dataloader.dataset.row_iteration_order
        total_outs = []
        for i in range(self.n_iters):
            np.random.seed(0)
            self.test_dataloader.dataset.row_iteration_order = row_iteration_order
            out_i = self._data_iter(iter_num=i, data=self.test_dataloader, model_mode=PREDICT,
                                    loss_func=loss_function,
                                    loss_func2=loss_function2)
            total_outs.append(out_i)
        return np.concatenate(total_outs, axis=1)
