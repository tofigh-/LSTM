from torch.utils.data import DataLoader
from utilities import append_lists
import time


class DatasetLoader(object):
    def __init__(self, dataset, mini_batch_size, num_workers, collate_fn=append_lists, num_db_calls_per_iter=1):
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False,
                                     batch_size=num_db_calls_per_iter)
        self.mini_batch_size = mini_batch_size

    def reshuffle_dataset(self):
        self.dataloader.dataset.reshuffle()

    def set_forecast_length(self, output_size):
        self.dataset.transform.set_forecast_length(output_size)

    def __iter__(self):
        st = time.time()
        for query_call_num, big_batch in enumerate(self.dataloader):
            print (time.time() - st)

            num_samples = len(big_batch)
            if num_samples % self.mini_batch_size != 0:
                num_mini_batchs = num_samples // self. \
                    mini_batch_size + 1
            else:
                num_mini_batchs = num_samples // self.mini_batch_size

            for idx in range(num_mini_batchs):
                yield big_batch[idx * self.mini_batch_size:(idx + 1) * self.mini_batch_size]
            st = time.time()
