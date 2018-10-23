import cPickle as pickle
import sqlite3
from random import shuffle

import numpy as np
from torch.utils.data import Dataset
import os
from subprocess import Popen
from os.path import join


class FileBasedDatasetReader(Dataset):

    def __init__(self, path_to_training_dir):
        self.path_to_training_dir = path_to_training_dir
        self.all_batch_files = sorted(filter(lambda x: x.endswith('_batch.npy'), os.listdir(self.path_to_training_dir)))
        self.all_loss_files = sorted(filter(lambda x: x.endswith('_loss.npy'), os.listdir(self.path_to_training_dir)))
        self.length = len(self.all_batch_files)
        self.temp_path = '/home/tnaghibi/cached_data'
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        process_main = Popen(['cp', join(path_to_training_dir, self.all_batch_files[0]), self.temp_path])
        Popen(['cp', join(path_to_training_dir, self.all_loss_files[0]), self.temp_path])
        process_main.wait()

    def __len__(self):
        return self.length

    def reset_length(self, length):
        self.length = length

    def reshuffle(self):
        return

    def __getitem__(self, idx):
        Popen(['cp', join(self.path_to_training_dir, self.all_batch_files[(idx + 1) % self.length]), self.temp_path])
        Popen(['cp', join(self.path_to_training_dir, self.all_loss_files[(idx + 1) % self.length]), self.temp_path])
        batch_data = np.load(os.path.join(self.temp_path, self.all_batch_files[idx]))
        loss_masks = np.load(os.path.join(self.temp_path, self.all_loss_files[idx]))
        Popen(['rm', join(self.temp_path, self.all_batch_files[idx])])
        Popen(['rm', join(self.temp_path, self.all_loss_files[idx])])
        return zip(batch_data, loss_masks)
