import cPickle as pickle
import sqlite3
from random import shuffle

import numpy as np
from torch.utils.data import Dataset
import os
from subprocess import Popen, PIPE
from os.path import join
from random import shuffle, seed
import signal
from data_accessor.data_loader.Settings import TOTAL_LENGTH


class FileBasedDatasetReader(Dataset):

    def __init__(self, path_to_training_dir):
        self.path_to_training_dir = path_to_training_dir
        self.all_batch_files = sorted(filter(lambda x: x.endswith('_batch.npy'), os.listdir(self.path_to_training_dir)))
        self.all_loss_files = sorted(filter(lambda x: x.endswith('_loss.npy'), os.listdir(self.path_to_training_dir)))
        self.length = len(self.all_batch_files)
        self.temp_path = '/home/tnaghibi/cached_data'
        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        self.idx_move = 50
        for i in range(self.idx_move):
            self._copy_(self.all_batch_files[i])
            self._copy_(self.all_loss_files[i])

    def _copy_(self, file_name, not_wait=False):

        cmd = ' '.join(['cp', join(self.path_to_training_dir, file_name), self.temp_path])
        if not_wait:
            cmd = cmd + " &"
        os.system(cmd)

    def _rm_(self, file_name, not_wait=False):
        cmd = ' '.join(['rm', join(self.temp_path, file_name)])
        if not_wait:
            cmd = cmd + " &"
        os.system(cmd)

    def __len__(self):
        return self.length

    def reset_length(self, length):
        self.length = length

    def reshuffle(self):
        seed_val = np.random.randint(1, 10000)
        seed(seed_val)
        shuffle(self.all_batch_files)
        seed(seed_val)
        shuffle(self.all_loss_files)

    def __getitem__(self, idx):
        if idx == self.idx_move:
            self._rm_("*")
            for i in range(50):
                self._copy_(self.all_batch_files[(idx + i) % self.length], not_wait=True)
                self._copy_(self.all_loss_files[(idx + i) % self.length], not_wait=True)
            self.idx_move = (self.idx_move + 50) % self.length

        batch_data = np.load(join(self.temp_path, self.all_batch_files[idx]))
        loss_masks = np.load(join(self.temp_path, self.all_loss_files[idx]))

        return zip(batch_data[:, -TOTAL_LENGTH:, :], loss_masks)
