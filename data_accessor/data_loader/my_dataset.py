import cPickle as pickle
import sqlite3
from random import shuffle

import numpy as np
from torch.utils.data import Dataset


class DatasetReader(Dataset):

    def __init__(self, path_to_training_db, transform=None, num_csku_per_query=10000, max_num_queries=None,
                 shuffle_dataset=True, seed=None,length_sort=True,
                 row_iteration_order=None):
        if row_iteration_order is not None and shuffle_dataset:
            raise ValueError('shuffle_dataset and row_iteration_order are mutually exclusive. '
                             'When row_iteration_order is provided, shuffle_dataset should be set to False.')
        if row_iteration_order is not None and seed is not None:
            raise ValueError('seed and row_iteration_order are mutually exclusive. '
                             'When row_iteration_order is provided, seed should be set to None.')

        self.length_sort = length_sort
        self.path_to_training_db = path_to_training_db
        self.big_batch_size = num_csku_per_query
        count_num_rows = 'SELECT max(_ROWID_) FROM data'
        self.training_db_file = path_to_training_db
        self.transform = transform
        connection = sqlite3.connect(path_to_training_db)
        conn_db = connection.cursor()
        conn_db.execute(count_num_rows)
        self.num_samples = conn_db.fetchall()[0][0]

        if shuffle_dataset:
            if seed is not None:
                np.random.seed(seed)
            self.row_iteration_order = np.random.choice(self.num_samples, self.num_samples, replace=False) + 1
        elif row_iteration_order is not None:
            self.row_iteration_order = row_iteration_order
            self.num_samples = len(row_iteration_order)
        else:
            self.row_iteration_order = np.arange(1, self.num_samples + 1)
        if max_num_queries is not None:
            self.length = max_num_queries
        else:
            self.length = self.num_samples // self.big_batch_size + 1
        connection.close()

    def query(self, row_indices):
        if len(row_indices) == 1:
            row_indices = list(row_indices)
            row_indices.append((row_indices[0]))
        return 'SELECT dictionary FROM data WHERE rowid IN {row_indices}'.format(row_indices=tuple(row_indices))

    def __len__(self):
        return self.length

    def reset_length(self, length):
        self.length = length

    def reshuffle(self):
        shuffle(self.row_iteration_order)

    def __getitem__(self, idx):
        try:
            connection = sqlite3.connect(self.training_db_file)
            connection_cursor = connection.cursor()
            row_range = np.take(range(self.num_samples),
                                range(idx * self.big_batch_size, (idx + 1) * self.big_batch_size),
                                mode='wrap')
            selected_samples = tuple(self.row_iteration_order[row_range])
            select_query = self.query(selected_samples)
            connection_cursor.execute(select_query)
            rows = connection_cursor.fetchall()
            selected_rows = []
            for row in rows:
                csku_object = pickle.loads(str(row[0]))
                if self.transform is not None:
                    csku_samples = self.transform(csku_object)
                    if csku_samples is not None and csku_samples != []:
                        selected_rows.extend(csku_samples)
                else:
                    selected_rows.append(csku_object)

        finally:
            connection.close()
            shuffle(selected_rows)
            if self.length_sort:
                self.transform.length_sort(selected_rows)
        return selected_rows
