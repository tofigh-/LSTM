import time
from os.path import join

from data_loader import DatasetLoader
from my_dataset import DatasetReader
from transformer import Transform
from utilities import append_lists

dir_path = "/backtesting_data"
file_name = "validation_data_Training_31a3e6e41cef24188ab2121d55be07ab98f7ccaf_2018-05-08_production.db"
validation_db = join(dir_path, file_name)


transform = Transform(db_file=validation_db,
                      min_start_date='2015-01-01',
                      max_end_date='2017-12-30',
                      training_transformation=True)

train_db = DatasetReader(path_to_training_db=validation_db,
                         transform=transform,
                         num_csku_per_query=1000,
                         shuffle_transform=True)

dataloader = DatasetLoader(train_db, collate_fn=append_lists, mini_batch_size=100, num_workers=0)
st = time.time()

for epoch in range(1):
    print "epoch %d" % epoch
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch % 99 ==0: print i_batch
        d = 1
    end_t = time.time()
    print i_batch,(end_t - st)
    st = time.time()

print (time.time() - st)
