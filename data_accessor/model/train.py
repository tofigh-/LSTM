from os.path import join

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from data_accessor.data_loader.Settings import *
from data_accessor.model.attention_transformer.prediciton import predict
from loss import L2Loss, L1Loss, LogNormalLoss, KPILoss
from model_utilities import exponential, cuda_converter, \
    kpi_compute_per_country, rounder

import sys

from data_accessor.model.attention_transformer.training import train_per_batch
from data_accessor.model.attention_transformer.predict_weight_update import predict_weight_update
from data_accessor.model.attention_transformer.encoder_decoder import EncoderDecoder


class Training(object):
    def __init__(self, model, train_dataloader, test_dataloader, validation_dataloader, n_iters):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.validation_dataloader = validation_dataloader
        self.n_iters = n_iters
        self.msloss = L2Loss(sum_loss=SUM_LOSS)
        self.l1loss = L1Loss(sum_loss=SUM_LOSS)

    def _kpi_print(self, mode, loss_value, kpi_value, weekly_aggregated_kpi, weekly_aggregated_kpi_scale, k1, k2,
                   predicted_country_sales=None, country_sales=None):
        print "{mode} Loss: {loss}".format(mode=mode, loss=loss_value)
        print "National {mode} Sale KPI {kpi}".format(mode=mode, kpi=kpi_value)

        print "Weekly {mode} Aggregated KPI {kpi}".format(mode=mode,
                                                          kpi=rounder(np.sum(weekly_aggregated_kpi, axis=0) / np.sum(
                                                              weekly_aggregated_kpi_scale, axis=0) * 100)
                                                          )
        global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(OUTPUT_SIZE)]
        print "National AVG {mode} KPI is {t_kpi}".format(mode=mode, t_kpi=global_kpi)
        if predicted_country_sales is not None:
            bias = [predicted_country_sales[i] / country_sales[i] for i in range(OUTPUT_SIZE)]
            print "Bias {mode} per country per week {bias}".format(mode=mode, bias=bias)

    def _mini_batch_preparation(self, batch_data):
        targets_future = dict()

        #  BATCH x OUTPUT x NUM_COUNTRY
        targets_future[SALES_MATRIX] = cuda_converter(torch.from_numpy(
            batch_data[:, TOTAL_INPUT:, feature_indices[SALES_MATRIX]].copy()
        ).float())

        targets_future[GLOBAL_SALE] = cuda_converter(torch.from_numpy(
            batch_data[:, TOTAL_INPUT:, feature_indices[GLOBAL_SALE][0]].copy()
        ).float())

        targets_future[STOCK] = cuda_converter(torch.from_numpy(
            batch_data[:, TOTAL_INPUT:, feature_indices[STOCK][0]].copy()
        ).float())
        batch_data[:, TOTAL_INPUT:, feature_indices[SALES_MATRIX]] = batch_data[:, TOTAL_INPUT - 1:TOTAL_INPUT,
                                                                     feature_indices[SALES_MATRIX]]
        batch_data[:, TOTAL_INPUT:, feature_indices[GLOBAL_SALE][0]] = batch_data[:, TOTAL_INPUT - 1:TOTAL_INPUT,
                                                                       feature_indices[GLOBAL_SALE][0]]
        batch_data[:, TOTAL_INPUT:, feature_indices[STOCK][0]] = batch_data[:, TOTAL_INPUT - 1:TOTAL_INPUT,
                                                                 feature_indices[STOCK][0]]
        black_price = exponential(
            cuda_converter(
                torch.from_numpy(
                    batch_data[:, TOTAL_INPUT - 1:TOTAL_INPUT, feature_indices[BLACK_PRICE_INT]]).float().squeeze()
            ),
            IS_LOG_TRANSFORM)
        return targets_future, batch_data, black_price

    def _data_iter(self, data, loss_func, loss_func2, teacher_forcing_ratio=1.0, train_mode=True):
        kpi_sale = [[] for _ in range(OUTPUT_SIZE)]
        kpi_sale_scale = [[] for _ in range(OUTPUT_SIZE)]
        weekly_aggregated_kpi = []
        weekly_aggregated_kpi_scale = []
        predicted_country_sales = [np.zeros(NUM_COUNTRIES) for _ in range(OUTPUT_SIZE)]
        country_sales = [np.zeros(NUM_COUNTRIES) for _ in range(OUTPUT_SIZE)]
        avg_loss = 0
        if train_mode:
            self.model.mode(train_mode=True)
        else:
            self.model.mode(train_mode=False)
        for batch_num, batch_data in enumerate(data):

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

            targets_future, batch_data, black_price = self._mini_batch_preparation(batch_data)
            if train_mode:

                loss, sale_predictions = train_per_batch(
                    model=self.model,
                    inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous(),
                    targets_future=targets_future,
                    loss_function=loss_func,
                    loss_function2=loss_func2,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                avg_loss = loss + avg_loss
                if batch_num % 100 == 0:
                    print "loss at num_batches {batch_number} is {loss_value}".format(batch_number=batch_num,
                                                                                      loss_value=loss)
                if batch_num % 100 == 0:
                    print "loss weights before update", self.model.loss_weights
                    self._update_loss_weights()
            else:
                loss, sale_predictions = predict(
                    model=self.model,
                    loss_function=loss_func,
                    loss_function2=loss_func2,
                    targets_future=targets_future,
                    inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous()
                )
                avg_loss = avg_loss + loss

            # Batch x Country
            weekly_aggregated = torch.sum(exponential(targets_future[SALES_MATRIX], IS_LOG_TRANSFORM),
                                          dim=1)
            weekly_aggregated_predictions = torch.sum(exponential(sale_predictions, IS_LOG_TRANSFORM),
                                                      dim=1)

            # size: (Country,)
            aggregated_err = torch.sum(torch.abs(weekly_aggregated - weekly_aggregated_predictions) * black_price,
                                       dim=0).data.cpu().numpy()
            aggregated_sale = torch.sum(weekly_aggregated * black_price, dim=0).data.cpu().numpy()

            weekly_aggregated_kpi.append(aggregated_err)
            weekly_aggregated_kpi_scale.append(aggregated_sale)
            if batch_num % 1000 == 0 and train_mode:
                weekly_aggregated_kpi_per_country = np.sum(np.array(weekly_aggregated_kpi), axis=0) / np.sum(
                    np.array(weekly_aggregated_kpi_scale), axis=0) * 100
                print "Weekly Aggregated Train KPI at Batch number {bn} is {kpi}".format(
                    bn=batch_num,
                    kpi=rounder(weekly_aggregated_kpi_per_country))

            for week_idx in range(OUTPUT_SIZE):
                target_sales = targets_future[SALES_MATRIX][:, week_idx, :]
                target_global_sales = targets_future[GLOBAL_SALE][:, week_idx]
                kpi_sale[week_idx].append(kpi_compute_per_country(sale_predictions[:, week_idx, :],
                                                                  target_sales=target_sales,
                                                                  target_global_sales=target_global_sales,
                                                                  log_transform=IS_LOG_TRANSFORM,
                                                                  weight=black_price
                                                                  ))
                predicted_country_sales[week_idx] = predicted_country_sales[week_idx] + torch.sum(
                    exponential(sale_predictions[:, week_idx, :], LOG_TRANSFORM), dim=0).data.cpu().numpy()

                real_sales = exponential(target_sales, IS_LOG_TRANSFORM)
                country_sales[week_idx] = country_sales[week_idx] + torch.sum(real_sales, dim=0).data.cpu().numpy()
                kpi_denominator = np.append(torch.sum(black_price * real_sales, dim=0).data.cpu().numpy(),
                                            torch.sum(real_sales * black_price).item())

                kpi_sale_scale[week_idx].append(kpi_denominator)
                if batch_num % 1000 == 0 and train_mode:
                    kpi_per_country = np.sum(np.array(kpi_sale[week_idx]), axis=0) / np.sum(
                        np.array(kpi_sale_scale[week_idx]),
                        axis=0) * 100

                    print "{i}ith week: National Train KPI at Batch number {bn} is {kpi}".format(
                        i=week_idx,
                        bn=batch_num,
                        kpi=rounder(kpi_per_country))
                    print "{i}ith week Natioanl AVG Train KPI is {t_kpi}".format(
                        i=week_idx,
                        t_kpi=np.sum(
                            np.array(kpi_sale[week_idx])[:, 0:-1]) / np.sum(
                            np.array(kpi_sale_scale[week_idx])[:,
                            0:-1]) * 100)
                    print "{i}th week bias is {bias}".format(i=week_idx,
                                                             bias=predicted_country_sales[week_idx] / country_sales[
                                                                 week_idx]
                                                             )

            if (batch_num + 1) % NUM_BATCH_SAVING_MODEL == 0 and train_mode:
                EncoderDecoder.save_checkpoint(self.model, 'attention_encoder_decoder.gz')

        kpi_per_country_total = [rounder(
            100 * np.sum(np.array(kpi_sale[i]), axis=0) / np.sum(np.array(kpi_sale_scale[i]), axis=0))
            for i in range(OUTPUT_SIZE)]
        return avg_loss / (batch_num + 1), np.array(kpi_sale), np.array(kpi_sale_scale), kpi_per_country_total, \
               predicted_country_sales, country_sales, np.array(weekly_aggregated_kpi), np.array(
            weekly_aggregated_kpi_scale)

    def train(self, resume=RESUME):
        if resume:
            EncoderDecoder.load_checkpoint({ENCODER_DECODER_CHECKPOINT: 'attention_encoder_decoder.gz'}, self.model)
        self.model.optimizer.zero_grad()
        np.random.seed(0)
        loss_function = self.msloss
        loss_function2 = self.l1loss
        for n_iter in range(1, self.n_iters + 1):
            print ("Iteration Number %d" % n_iter)

            change_optimizer_epoch = 1
            if RESUME:
                self.model.mode(train_mode=False)
                test_loss, k1, k2, test_sale_kpi, \
                predicted_country_sales_test, \
                country_sales_test, \
                weekly_aggregated_kpi_test, \
                weekly_aggregated_kpi_scale_test = self._data_iter(
                    data=self.test_dataloader,
                    train_mode=False,
                    loss_func=loss_function,
                    loss_func2=loss_function2
                )
                self.model.mode(train_mode=True)
                self._kpi_print("Test", test_loss, test_sale_kpi, weekly_aggregated_kpi_test,
                                weekly_aggregated_kpi_scale_test, k1, k2, predicted_country_sales_test,
                                country_sales_test)

            if n_iter == change_optimizer_epoch:
                scheduler = CosineAnnealingLR(self.model.optimizer.optimizer, T_max=6, eta_min=0.000001)

            train_loss, _, _, \
            train_sale_kpi, \
            predicted_country_sales, \
            country_sales, weekly_aggregated_kpi, \
            weekly_aggregated_kpi_scale = self._data_iter(
                data=self.train_dataloader,
                train_mode=True,
                loss_func=loss_function,
                loss_func2=loss_function2,
                teacher_forcing_ratio=False,
            )
            print "National Train Sale KPI {kpi}".format(kpi=train_sale_kpi)
            print "Weekly Aggregated KPI {kpi}".format(
                kpi=rounder(np.sum(weekly_aggregated_kpi, axis=0) / np.sum(weekly_aggregated_kpi_scale, axis=0) * 100)
            )
            EncoderDecoder.save_checkpoint(self.model, 'attention_encoder_decoder.gz')

            self.train_dataloader.reshuffle_dataset()

            self.model.mode(train_mode=False)
            test_loss, k1, k2, test_sale_kpi, \
            predicted_country_sales_test, \
            country_sales_test, \
            weekly_aggregated_kpi_test, \
            weekly_aggregated_kpi_scale_test = self._data_iter(
                data=self.test_dataloader,
                train_mode=False,
                loss_func=loss_function,
                loss_func2=loss_function2
            )
            self.model.mode(train_mode=True)
            self._kpi_print("Test", test_loss, test_sale_kpi, weekly_aggregated_kpi_test,
                            weekly_aggregated_kpi_scale_test, k1, k2, predicted_country_sales_test,
                            country_sales_test)

            if n_iter > change_optimizer_epoch:
                scheduler.step(n_iter)

        self.model.mode(train_mode=False)
        test_loss, k1, k2, test_sale_kpi, \
        predicted_country_sales_test, \
        country_sales_test, \
        weekly_aggregated_kpi_test, \
        weekly_aggregated_kpi_scale_test = self._data_iter(self.test_dataloader, train_mode=False,
                                                           loss_func=loss_function,
                                                           loss_func2=loss_function2)
        self._kpi_print("Test", test_loss, test_sale_kpi, weekly_aggregated_kpi_test,
                        weekly_aggregated_kpi_scale_test, k1, k2, predicted_country_sales_test,
                        country_sales_test)

    def _update_loss_weights(self):
        kpi_loss = KPILoss()
        self.model.mode(train_mode=True)
        avg_loss = 0
        loss_l2_avg = cuda_converter(torch.zeros(len(list_l2_loss_countries)))
        loss_l1_avg = cuda_converter(torch.zeros(len(list_l1_loss_countries)))
        all_param_grads = []
        for idx, param in enumerate(self.model.parameters()):
            all_param_grads.append(torch.empty([NUM_COUNTRIES] + list(param.shape)))

        for batch_num, batch_data in enumerate(self.validation_dataloader):
            batch_data = np.array(batch_data)
            targets_future, batch_data, black_price = self._mini_batch_preparation(batch_data)
            loss_l2_countries, loss_l1_countries, loss_kpi = predict_weight_update(
                model=self.model,
                loss_function=self.msloss,
                loss_function2=self.l1loss,
                targets_future=targets_future,
                inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous(),
                update_weights_mode=True,
                kpi_loss=kpi_loss,
                weights=black_price
            )
            avg_loss = avg_loss + loss_kpi
            loss_l1_avg += loss_l1_countries
            loss_l2_avg += loss_l2_countries

        kpi_loss_grads = []

        avg_loss.backward(retain_graph=True)
        loss_weight_gradients = cuda_converter(torch.zeros(NUM_COUNTRIES))
        for param in self.model.parameters():
            kpi_loss_grads.append(param.grad)
            param.grad = None
        for loss, country_id in zip(loss_l2_avg, list_l2_loss_countries):
            loss.backward(retain_graph=True)
            for idx, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    loss_weight_gradients[country_id] += (-param.grad * kpi_loss_grads[idx]).mean()
                param.grad = None
        for loss, country_id in zip(loss_l1_avg, list_l1_loss_countries):
            loss.backward(retain_graph=True)
            for idx, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    loss_weight_gradients[country_id] += (-param.grad * kpi_loss_grads[idx]).mean()
                param.grad = None

        for country_id in range(NUM_COUNTRIES):
            self.model.loss_weights[country_id].grad = cuda_converter(torch.tensor([loss_weight_gradients[country_id]]))
        self.model.loss_weight_optimizer.step()
        self.model.loss_weight_optimizer.zero_grad()
        sum_value = 0
        for country_id in range(NUM_COUNTRIES):
            self.model.loss_weights[country_id] = torch.clamp(self.model.loss_weights[country_id], min=0)
            sum_value += self.model.loss_weights[country_id]
        for country_id in range(NUM_COUNTRIES):
            self.model.loss_weights[country_id] = self.model.loss_weights[country_id] / sum_value
        # self._loss_weight_mode(train_mode=False)
        self.model.optimizer.zero_grad()
        print "average loss for updating weights", loss_weight_gradients
        print "loss weights after update", self.model.loss_weights
