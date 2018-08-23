from os.path import join

import numpy as np
import torch

from data_accessor.data_loader.Settings import *
from data_accessor.data_loader.data_loader import DatasetLoader
from data_accessor.data_loader.my_dataset import DatasetReader
from data_accessor.data_loader.my_feature_class import MyFeatureClass
from data_accessor.data_loader.transformer import Transform
from loss import L2_LOSS, L1_LOSS
from model_utilities import kpi_compute, exponential, complete_embedding_description, cuda_converter, \
    kpi_compute_per_country, rounder
from data_accessor.data_loader.utilities import load_label_encoder, save_label_encoder
from vanilla_rnn import VanillaRNNModel
import os
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

dir_path = ""
file_name = "training.db"
label_encoder_file = "label_encoders.json"
validation_db = join(dir_path, file_name)
debug_mode = False
if debug_mode:
    num_csku_per_query_train = 50
    num_csku_per_query_test = 10
    max_num_queries_train = 1
    max_num_queries_test = 1
    num_workers = 0
else:
    num_csku_per_query_train = 5000
    num_csku_per_query_test = 10000
    max_num_queries_train = None
    max_num_queries_test = 4
    num_workers = 4

if os.path.exists(label_encoder_file):
    label_encoders = load_label_encoder(label_encoder_file)
else:
    label_encoders = None
my_feature_class_train = MyFeatureClass(FEATURE_DESCRIPTIONS, low_sale_percentage=1.0)
train_transform = Transform(
    feature_transforms=my_feature_class_train,
    label_encoders=label_encoders,
    db_file=validation_db,
    min_start_date='2014-01-01',
    max_end_date='2016-12-28',
    training_transformation=True,
    keep_zero_stock_filter=0.5)
if label_encoders is None:
    label_encoders = train_transform.label_encoders
    save_label_encoder(label_encoders, label_encoder_file)
    print ("Saving Label Encoder Done.")

my_feature_class_test = MyFeatureClass(FEATURE_DESCRIPTIONS, low_sale_percentage=1.0)
test_transform = Transform(
    feature_transforms=my_feature_class_test,
    label_encoders=label_encoders,
    db_file=validation_db,
    target_dates=['2017-01-07'],
    training_transformation=True,
    keep_zero_stock_filter=1.0,
    activate_filters=True)

train_db = DatasetReader(
    path_to_training_db=validation_db,
    transform=train_transform,
    num_csku_per_query=num_csku_per_query_train,
    max_num_queries=max_num_queries_train,
    shuffle_dataset=True)

test_db = DatasetReader(
    path_to_training_db=validation_db,
    transform=test_transform,
    num_csku_per_query=num_csku_per_query_test,
    max_num_queries=max_num_queries_test,
    shuffle_dataset=True,
    seed=42)
train_dataloader = DatasetLoader(train_db, mini_batch_size=BATCH_SIZE, num_workers=num_workers)
test_dataloader = DatasetLoader(test_db, mini_batch_size=TEST_BATCH_SIZE, num_workers=0)
embedding_descripts = complete_embedding_description(embedding_descriptions, label_encoders)
vanilla_rnn = VanillaRNNModel(embedding_descripts,
                              load_saved_model=False,
                              is_attention=False,
                              num_output=NUM_COUNTRIES)


def train(vanilla_rnn, n_iters, resume=RESUME):
    if resume:
        vanilla_rnn.load_checkpoint({FUTURE_DECODER_CHECKPOINT: 'decoder.gz', ENCODER_CHECKPOINT: 'encoder.gz'})
    vanilla_rnn.encoder_optimizer.zero_grad()
    vanilla_rnn.future_decoder_optimizer.zero_grad()
    msloss = L2_LOSS(size_average=SIZE_AVERAGE, sum_weight=SUM_WEIGHT)
    l1loss = L1_LOSS(size_average=SIZE_AVERAGE, sum_weight=SUM_WEIGHT)
    loss_function = msloss
    np.random.seed(0)

    def data_iter(data, loss_func, loss_func2, teacher_forcing_ratio=1.0, train_mode=True):
        kpi_sale = [[] for _ in range(OUTPUT_SIZE)]
        kpi_sale_scale = [[] for _ in range(OUTPUT_SIZE)]
        if train_mode: vanilla_rnn.mode(train_mode=True)
        for batch_num, batch_data in enumerate(data):

            if batch_num % 10001 == 0 and train_mode:
                vanilla_rnn.mode(train_mode=False)
                k1, k2, test_sale_kpi = data_iter(data=test_dataloader, train_mode=False, loss_func=loss_function,
                                                  loss_func2=loss_func2)
                vanilla_rnn.mode(train_mode=True)
                print "National Test Sale KPI {kpi}".format(kpi=test_sale_kpi)
                global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(OUTPUT_SIZE)]
                print "Global Test KPI is {t_kpi}".format(t_kpi=global_kpi)

            batch_data = np.swapaxes(np.array(batch_data), axis1=0, axis2=1)
            input_encode = cuda_converter(torch.from_numpy(batch_data[0:TOTAL_INPUT, :, :]).float()).contiguous()
            input_decode = cuda_converter(
                torch.from_numpy(batch_data[TOTAL_INPUT:TOTAL_LENGTH, :, :]).float()).contiguous()
            targets_future = dict()

            # OUT_PUT x BATCH x NUM_COUNTRY
            targets_future[SALES_MATRIX] = input_decode[:, :, feature_indices[SALES_MATRIX]].clone() + input_decode[:, :, feature_indices[PAST_MEAN_SALE]].clone()
            targets_future[GLOBAL_SALE] = input_decode[:, :, feature_indices[GLOBAL_SALE][0]].clone()

            targets_future[STOCK] = input_decode[:, :, feature_indices[STOCK][0]].clone()

            input_decode[:, :, feature_indices[SALES_MATRIX]] = input_encode[-1, :, feature_indices[SALES_MATRIX]]
            input_decode[:, :, feature_indices[GLOBAL_SALE][0]] = input_encode[-1, :, feature_indices[GLOBAL_SALE][0]]
            input_decode[:, :, feature_indices[STOCK][0]] = input_encode[-1, :, feature_indices[STOCK][0]]
            input_decode[:, :, feature_indices[PAST_MEAN_SALE]] = input_encode[-1, :, feature_indices[PAST_MEAN_SALE]]
            black_price = exponential(input_encode[-1, :, feature_indices[BLACK_PRICE_INT]], IS_LOG_TRANSFORM)
            if train_mode:
                loss, output_global_sale, sale_predictions = vanilla_rnn.train(
                    inputs=(input_encode, input_decode),
                    targets_future=targets_future,
                    loss_function=loss_func,
                    loss_function2=loss_func2,
                    teacher_forcing_ratio=1.0
                )
                if batch_num % 100 == 0:
                    print "loss at num_batches {batch_number} is {loss_value}".format(batch_number=batch_num,
                                                                                      loss_value=loss)
            else:
                # TODO to generalize KPI computation to many weeks this 0 should go away
                output_global_sale, sale_predictions = vanilla_rnn.predict_over_period(
                    inputs=(input_encode, input_decode))
            for i in range(OUTPUT_SIZE):
                target_sales = targets_future[SALES_MATRIX][i, :, :]
                target_global_sales = targets_future[GLOBAL_SALE][i, :]
                kpi_sale[i].append(kpi_compute_per_country(sale_predictions[i],
                                                           target_sales=target_sales,
                                                           target_global_sales=target_global_sales,
                                                           log_transform=IS_LOG_TRANSFORM,
                                                           weight=black_price
                                                           ))
                real_sales = exponential(targets_future[SALES_MATRIX][i, :, :], IS_LOG_TRANSFORM)
                kpi_denominator = np.append(torch.sum(black_price * real_sales, dim=0).data.cpu().numpy(),
                                            torch.sum(real_sales * black_price).item())

                kpi_sale_scale[i].append(kpi_denominator)
                if batch_num % 1000 == 0 and train_mode:
                    kpi_per_country = np.sum(np.array(kpi_sale[i]), axis=0) / np.sum(np.array(kpi_sale_scale[i]),
                                                                                     axis=0) * 100
                    print "{i}ith week: National Train KPI at Batch number {bn} is {kpi}".format(
                        i=i,
                        bn=batch_num,
                        kpi=rounder(kpi_per_country))
                    print "{i}ith week Global Train KPI is {t_kpi}".format(
                        i=i,
                        t_kpi=np.sum(
                            np.array(kpi_sale[i])[:, 0:-1]) / np.sum(
                            np.array(kpi_sale_scale[i])[:,
                            0:-1]) * 100)

            if (batch_num + 1) % NUM_BATCH_SAVING_MODEL == 0 and train_mode:
                vanilla_rnn.save_checkpoint(encoder_file_name='encoder.gz', future_decoder_file_name='decoder.gz')

        kpi_per_country_total = [rounder(
            100 * np.sum(np.array(kpi_sale[i]), axis=0) / np.sum(np.array(kpi_sale_scale[i]), axis=0))
            for i in range(OUTPUT_SIZE)]
        return np.array(kpi_sale), np.array(kpi_sale_scale), kpi_per_country_total

    for n_iter in range(1, n_iters + 1):
        print ("Iteration Number %d" % n_iter)
        if n_iter <= 3:
            teacher_forcing_ratio = 1.0
        else:
            teacher_forcing_ratio = 0.3
        if n_iter <= 4:
            loss_function = msloss
            loss_function2 = loss_function
        else:
            loss_function = l1loss
            loss_function2 = msloss
        _, _, train_sale_kpi = data_iter(data=train_dataloader,
                                         train_mode=True,
                                         loss_func=loss_function,
                                         loss_func2=loss_function2,
                                         teacher_forcing_ratio=teacher_forcing_ratio)
        print "Train Sale KPI {kpi}".format(kpi=train_sale_kpi)
        vanilla_rnn.save_checkpoint(encoder_file_name='encoder.gz', future_decoder_file_name='decoder.gz')

        train_dataloader.reshuffle_dataset()

    vanilla_rnn.mode(train_mode=False)
    k1, k2, test_sale_kpi = data_iter(test_dataloader, train_mode=False, loss_func=loss_function,
                                      loss_func2=loss_function2)
    print "National Test Sale KPI {kpi}".format(kpi=test_sale_kpi)
    global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(OUTPUT_SIZE)]
    print "Global Test KPI is {t_kpi}".format(t_kpi=global_kpi)


train(vanilla_rnn, n_iters=8)
