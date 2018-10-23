import numpy as np
from numpy import save as np_save
from torch.utils.data import DataLoader
from data_accessor.data_loader.utilities import append_lists
from data_accessor.data_loader.Settings import *
from time import time
from random import shuffle, seed
import os
import sys

def cache_data(dataset):
    extension = 'npy'
    dir_path = '/data/cached_data'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    def save_batches(all_losses, all_batches, batch_num):
        st = time()
        seed(42)
        shuffle(all_batches)
        seed(42)
        shuffle(all_losses)
        output_path_batch = os.path.join(dir_path,
                                         "chuck_saved_{seq_length}_{output_size}_{batch_num}_batch.{extension}".format(
                                             batch_num=batch_num,
                                             seq_length=TOTAL_INPUT,
                                             output_size=OUTPUT_SIZE,
                                             extension=extension))
        output_path_loss = os.path.join(dir_path,
                                        "chuck_saved_{seq_length}_{output_size}_{batch_num}_loss.{extension}".format(
                                            batch_num=batch_num,
                                            seq_length=TOTAL_INPUT,
                                            output_size=OUTPUT_SIZE,
                                            extension=extension)
                                        )
        save(np.concatenate(all_batches), output_path_batch)
        print ("time to save batch {time_amount} sec".format(time_amount=time() - st))
        save(np.concatenate(all_losses), output_path_loss)

    dataloader = DataLoader(dataset, num_workers=0, collate_fn=append_lists, pin_memory=False,
                            batch_size=1)
    all_losses = []
    all_batches = []
    for batch_num, large_batch in enumerate(dataloader):

        print batch_num
        if batch_num == 0: print ("length of one batch data is {length_b}".format(length_b=len(large_batch)))
        loss_masks = []
        batch_data = []
        batch_data[:], loss_masks[:] = zip(*large_batch)
        batch_data = np.array(batch_data)
        loss_masks = np.array(loss_masks)
        all_batches.append(batch_data)
        all_losses.append(loss_masks)
        if (batch_num + 1) % 5 == 0:
            save_batches(all_losses, all_batches, batch_num)
            all_losses = []
            all_batches = []
        sys.stdout.flush()

    save_batches(all_losses, all_batches, batch_num + 1)
    all_losses = []
    all_batches = []


def save(arr, file_path):
    with open(file_path, 'wb+') as output_file:
        np_save(output_file, arr, allow_pickle=False)
        output_file.flush()
