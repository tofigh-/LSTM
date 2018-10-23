import cPickle as pickle
import sqlite3
from random import shuffle

import numpy as np
from torch.utils.data import Dataset
import os


class FileBasedDatasetReader(Dataset):

    def __init__(self, path_to_training_dir):
        self.path_to_training_dir = path_to_training_dir
        self.all_batch_files = sorted(filter(lambda x: x.endswith('_batch.npy'), os.listdir(self.path_to_training_dir)))
        self.all_loss_files = sorted(filter(lambda x: x.endswith('_loss.npy'), os.listdir(self.path_to_training_dir)))
        self.length = len(self.all_batch_files)

    def __len__(self):
        return self.length

    def reset_length(self, length):
        self.length = length

    def reshuffle(self):
        return

    def __getitem__(self, idx):
        batch_data = np.load(os.path.join(self.path_to_training_dir, self.all_batch_files[idx]))
        loss_masks = np.load(os.path.join(self.path_to_training_dir, self.all_loss_files[idx]))
        return zip(batch_data, loss_masks)
