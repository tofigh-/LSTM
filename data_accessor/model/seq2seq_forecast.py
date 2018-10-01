from os.path import join

import torch

from data_accessor.data_loader.Settings import *
from data_accessor.data_loader.data_loader import DatasetLoader
from data_accessor.data_loader.my_dataset import DatasetReader
from data_accessor.data_loader.my_feature_class import MyFeatureClass
from data_accessor.data_loader.transformer import Transform
from loss import L2_LOSS, L1_LOSS, LogNormalLoss
from model_utilities import exponential, complete_embedding_description, cuda_converter, \
    kpi_compute_per_country, rounder
from data_accessor.data_loader.utilities import load_label_encoder, save_label_encoder
from vanilla_rnn import VanillaRNNModel
import sys
import os
from data_accessor.data_loader import Settings as settings
from datetime import datetime
from datetime import timedelta
import git
from data_accessor.model.attention_transformer.attention_transformer_model import make_model
from data_accessor.model.attention_transformer.training import train_per_batch
from data_accessor.model.attention_transformer.prediciton import predict
from data_accessor.model.attention_transformer.encoder_decoder import EncoderDecoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
for variable in to_print_variables:
    print (variable, settings.__dict__[variable])
repo = git.Repo(search_parent_directories=True)
commit_hash = repo.head.object.hexsha
branch_name = repo.active_branch.path
print "commit_hash: " + commit_hash
print "branch_name: " + branch_name

dir_path = ""
file_name = "training.db"
label_encoder_file = "label_encoders.json"
validation_db = join(dir_path, file_name)
debug_mode = False


if debug_mode:
    num_csku_per_query_train = 500
    num_csku_per_query_test = 100
    train_workers = 0
    max_num_queries_train = 1
    max_num_queries_test = 1
else:
    num_csku_per_query_train = 10000
    train_workers = 4
    num_csku_per_query_test = 10000
    max_num_queries_train = None
    max_num_queries_test = 5

if os.path.exists(label_encoder_file):
    label_encoders = load_label_encoder(label_encoder_file)
else:
    label_encoders = None
my_feature_class_train = MyFeatureClass(FEATURE_DESCRIPTIONS, low_sale_percentage=1.0)
max_end_date = datetime.strptime('2016-12-28', '%Y-%m-%d').date()
target_test_date = max_end_date + timedelta(weeks=OUTPUT_SIZE + 1)
train_transform = Transform(
    feature_transforms=my_feature_class_train,
    label_encoders=label_encoders,
    db_file=validation_db,
    min_start_date='2014-01-01',
    max_end_date=max_end_date,
    training_transformation=True,
    keep_zero_stock_filter=0.0,
    keep_percentage_zero_price=0.0,
    stock_threshold=TRAIN_STOCK_THRESHOLD,
    keep_zero_sale_filter=0.1,
    activate_filters=True)

if label_encoders is None:
    label_encoders = train_transform.label_encoders
    save_label_encoder(label_encoders, label_encoder_file)
    print ("Saving Label Encoder Done.")

my_feature_class_test = MyFeatureClass(FEATURE_DESCRIPTIONS, low_sale_percentage=1.0)
test_transform = Transform(
    feature_transforms=my_feature_class_test,
    label_encoders=label_encoders,
    db_file=validation_db,
    target_dates=[target_test_date],
    training_transformation=True,
    keep_zero_stock_filter=0.0,
    keep_percentage_zero_price=0.0,
    stock_threshold=TEST_STOCK_THRESHOLD,
    keep_zero_sale_filter=1.0,
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
train_dataloader = DatasetLoader(train_db, mini_batch_size=BATCH_SIZE, num_workers=train_workers)
test_dataloader = DatasetLoader(test_db, mini_batch_size=TEST_BATCH_SIZE, num_workers=0)
embedding_descripts = complete_embedding_description(embedding_descriptions, label_encoders)
vanilla_rnn = VanillaRNNModel(embedding_descripts,
                              load_saved_model=False,
                              is_attention=False,
                              num_output=NUM_COUNTRIES)

d_model = len(numeric_feature_indices)
print "d_model is: " + str(d_model)
attention_model = cuda_converter(make_model(embedding_descriptions=embedding_descripts,
                                            total_input=TOTAL_INPUT,
                                            forecast_length=OUTPUT_SIZE,
                                            N=3,
                                            d_model=d_model,
                                            d_ff=4 * 96,
                                            h=14,
                                            dropout_enc=0.1,
                                            dropout_dec=0.1))


def train(attention_model, n_iters, resume=RESUME):
    if resume:
        EncoderDecoder.load_checkpoint({ENCODER_DECODER_CHECKPOINT: 'attention_encoder_decoder.gz'}, attention_model,
                                       attention_model.optimizer)
    attention_model.optimizer.zero_grad()

    msloss = L2_LOSS(size_average=SIZE_AVERAGE, sum_weight=SUM_WEIGHT)
    l1loss = L1_LOSS(size_average=SIZE_AVERAGE, sum_weight=SUM_WEIGHT)
    lognormal_loss = LogNormalLoss(size_average=SIZE_AVERAGE)
    np.random.seed(0)

    def data_iter(data, loss_func, loss_func2, teacher_forcing_ratio=1.0, train_mode=True):
        kpi_sale = [[] for _ in range(OUTPUT_SIZE)]
        kpi_sale_scale = [[] for _ in range(OUTPUT_SIZE)]
        weekly_aggregated_kpi = []
        weekly_aggregated_kpi_scale = []
        predicted_country_sales = [np.zeros(NUM_COUNTRIES) for _ in range(OUTPUT_SIZE)]
        country_sales = [np.zeros(NUM_COUNTRIES) for _ in range(OUTPUT_SIZE)]

        if train_mode: attention_model.mode(train_mode=True)

        for batch_num, batch_data in enumerate(data):

            if batch_num % 10001 == 0 and train_mode:
                attention_model.mode(train_mode=False)
                k1, k2, test_sale_kpi, \
                predicted_country_sales_test, \
                country_sales_test, \
                weekly_aggregated_kpi_test, \
                weekly_aggregated_kpi_scale_test = data_iter(
                    data=test_dataloader,
                    train_mode=False,
                    loss_func=loss_func,
                    loss_func2=loss_func2
                )
                attention_model.mode(train_mode=True)

                print "National Test Sale KPI {kpi}".format(kpi=test_sale_kpi)
                print "Weekly Test Aggregated KPI {kpi}".format(
                    kpi=rounder(
                        np.sum(weekly_aggregated_kpi_test, axis=0) / np.sum(weekly_aggregated_kpi_scale_test,
                                                                            axis=0) * 100)
                )
                global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(OUTPUT_SIZE)]
                print "Natioanl Test KPI is {t_kpi}".format(t_kpi=global_kpi)
                bias = [predicted_country_sales_test[i] / country_sales_test[i] for i in range(OUTPUT_SIZE)]
                print "Bias Test per country per week {bias}".format(bias=bias)

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

            if train_mode:

                loss, sale_predictions = train_per_batch(
                    model=attention_model,
                    inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous(),
                    targets_future=targets_future,
                    loss_function=loss_func,
                    loss_function2=loss_func2,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                if batch_num % 100 == 0:
                    print "loss at num_batches {batch_number} is {loss_value}".format(batch_number=batch_num,
                                                                                      loss_value=loss)
            else:
                sale_predictions = predict(
                    model=attention_model,
                    inputs=cuda_converter(torch.from_numpy(batch_data).float()).contiguous()
                )
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
                EncoderDecoder.save_checkpoint(attention_model, 'attention_encoder_decoder.gz')

        kpi_per_country_total = [rounder(
            100 * np.sum(np.array(kpi_sale[i]), axis=0) / np.sum(np.array(kpi_sale_scale[i]), axis=0))
            for i in range(OUTPUT_SIZE)]
        return np.array(kpi_sale), np.array(kpi_sale_scale), kpi_per_country_total, \
               predicted_country_sales, country_sales, np.array(weekly_aggregated_kpi), np.array(
            weekly_aggregated_kpi_scale)

    for n_iter in range(1, n_iters + 1):
        print ("Iteration Number %d" % n_iter)
        loss_function = lognormal_loss
        loss_function2 = msloss
        if n_iter <= 1:
            teacher_forcing_ratio = 0.0
        else:
            teacher_forcing_ratio = 0.0

        _, _, \
        train_sale_kpi, \
        predicted_country_sales, \
        country_sales, weekly_aggregated_kpi, \
        weekly_aggregated_kpi_scale = data_iter(data=train_dataloader,
                                                train_mode=True,
                                                loss_func=loss_function,
                                                loss_func2=loss_function2,
                                                teacher_forcing_ratio=teacher_forcing_ratio,
                                                )
        print "National Train Sale KPI {kpi}".format(kpi=train_sale_kpi)
        print "Weekly Aggregated KPI {kpi}".format(
            kpi=rounder(np.sum(weekly_aggregated_kpi, axis=0) / np.sum(weekly_aggregated_kpi_scale, axis=0) * 100)
        )
        EncoderDecoder.save_checkpoint(attention_model, 'attention_encoder_decoder.gz')

        train_dataloader.reshuffle_dataset()

    attention_model.mode(train_mode=False)
    k1, k2, test_sale_kpi, \
    predicted_country_sales, \
    country_sales, \
    weekly_aggregated_kpi, \
    weekly_aggregated_kpi_scale = data_iter(test_dataloader, train_mode=False,
                                            loss_func=loss_function,
                                            loss_func2=loss_function2)

    print "National Test Sale KPI {kpi}".format(kpi=test_sale_kpi)

    print "Weekly Test Aggregated KPI {kpi}".format(
        kpi=rounder(np.sum(weekly_aggregated_kpi, axis=0) / np.sum(weekly_aggregated_kpi_scale, axis=0) * 100)
    )
    global_kpi = [np.sum(k1[i, :, 0:-1]) / np.sum(k2[i, :, 0:-1]) * 100 for i in range(OUTPUT_SIZE)]
    bias = [predicted_country_sales[i] / country_sales[i] for i in range(OUTPUT_SIZE)]
    print "National AVG Test KPI is {t_kpi}".format(t_kpi=global_kpi)
    print "Bias Test per country per week {bias}".format(bias=bias)


train(attention_model, n_iters=50)
