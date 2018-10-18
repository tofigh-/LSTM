from os.path import join
import copy

from data_accessor.data_loader.Settings import *
from data_accessor.data_loader.data_loader import DatasetLoader
from data_accessor.data_loader.my_dataset import DatasetReader
from data_accessor.data_loader.my_feature_class import MyFeatureClass
from data_accessor.data_loader.transformer import Transform
from model_utilities import complete_embedding_description, cuda_converter
from data_accessor.data_loader.utilities import load_label_encoder, save_label_encoder
import sys
import os
from data_accessor.data_loader import Settings as settings
from datetime import datetime
from datetime import timedelta
import git
from data_accessor.model.attention_transformer.attention_transformer_model import make_model
from train import Training
import sqlite3

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
path_to_training_db = join(dir_path, file_name)
debug_mode = False

if debug_mode:
    num_csku_per_query_train = 500
    num_csku_per_query_test = 100
    num_csku_per_query_validation = 50
    train_workers = 0
    max_num_queries_train = 1
    max_num_queries_test = 1
    max_num_queries_validation = 1

else:
    num_csku_per_query_train = 10000
    num_csku_per_query_validation = 1000
    train_workers = 4
    num_csku_per_query_test = 10000
    max_num_queries_train = None
    max_num_queries_test = 5
    max_num_queries_validation = 1

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
    db_file=path_to_training_db,
    min_start_date='2014-01-01',
    max_end_date=max_end_date,
    training_transformation=True,
    keep_zero_stock_filter=0.0,
    keep_percentage_zero_price=0.0,
    stock_threshold=TRAIN_STOCK_THRESHOLD,
    keep_zero_sale_filter=TRAIN_ZERO_SALE_PERCENTAGE,
    activate_filters=True)

validation_transform = copy.deepcopy(train_transform)
validation_transform.stock_threshold = TEST_STOCK_THRESHOLD
validation_transform.keep_zero_sale_filter = TEST_ZERO_SALE_PERCENTAGE

if label_encoders is None:
    label_encoders = train_transform.label_encoders
    save_label_encoder(label_encoders, label_encoder_file)
    print ("Saving Label Encoder Done.")

my_feature_class_test = MyFeatureClass(FEATURE_DESCRIPTIONS, low_sale_percentage=1.0)
test_transform = Transform(
    feature_transforms=my_feature_class_test,
    label_encoders=label_encoders,
    db_file=path_to_training_db,
    target_dates=[target_test_date],
    training_transformation=True,
    keep_zero_stock_filter=0.0,
    keep_percentage_zero_price=0.0,
    stock_threshold=TEST_STOCK_THRESHOLD,
    keep_zero_sale_filter=TEST_ZERO_SALE_PERCENTAGE,
    activate_filters=True)


train_db = DatasetReader(
    path_to_training_db=path_to_training_db,
    transform=train_transform,
    num_csku_per_query=num_csku_per_query_train,
    max_num_queries=max_num_queries_train,
    shuffle_dataset=True)

validation_db = DatasetReader(
    path_to_training_db=path_to_training_db,
    transform=validation_transform,
    num_csku_per_query=num_csku_per_query_validation,
    max_num_queries=max_num_queries_validation,
    shuffle_dataset=True)

test_db = DatasetReader(
    path_to_training_db=path_to_training_db,
    transform=test_transform,
    num_csku_per_query=num_csku_per_query_test,
    max_num_queries=max_num_queries_test,
    shuffle_dataset=True,
    seed=42)
train_dataloader = DatasetLoader(train_db, mini_batch_size=BATCH_SIZE, num_workers=train_workers)
validation_dataloader = DatasetLoader(validation_db, mini_batch_size=TEST_BATCH_SIZE, num_workers=0)
test_dataloader = DatasetLoader(test_db, mini_batch_size=TEST_BATCH_SIZE, num_workers=0)
embedding_descripts = complete_embedding_description(embedding_descriptions, label_encoders)

d_model = len(numeric_feature_indices)
print "d_model is: " + str(d_model)
attention_model = cuda_converter(make_model(embedding_descriptions=embedding_descripts,
                                            total_input=TOTAL_INPUT,
                                            forecast_length=OUTPUT_SIZE,
                                            N=1,
                                            d_model=d_model,
                                            d_ff=4 * d_model,
                                            h=14,
                                            dropout_enc=0.1,
                                            dropout_dec=0.1))

training = Training(model=attention_model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                    validation_dataloader=validation_dataloader, n_iters=50)
training.train(resume=RESUME)
