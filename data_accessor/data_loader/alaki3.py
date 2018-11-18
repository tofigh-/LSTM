import sqlite3
import cPickle as pickle
import cPickle as pickle
from Settings import *

training_db_file = "/Users/tnaghibi/PycharmProjects/data_accessor/model/training.db"

COMMODITY_GROUPS = 'Commodity Groups'
CATEGORY = 'Category'


def compute_cg_dict(db_file, output_file, chunk_size=1000):
    connection = sqlite3.connect(db_file)
    conn_db = connection.cursor()
    count_num_rows = 'SELECT max(_ROWID_) FROM data'
    conn_db.execute(count_num_rows)
    num_samples = conn_db.fetchall()[0][0]
    row_indices = range(1, num_samples + 1)
    query = lambda row_indices: 'SELECT dictionary FROM data WHERE rowid IN {row_indices}'.format(
        row_indices=tuple(row_indices))
    # TODO
    num_chunks = num_samples // chunk_size + 1
    print "total_num_chunks", num_chunks
    # num_chunks = 1
    dict_cgs = {}
    for i in range(num_chunks):
        selected_samples = tuple(row_indices[i * chunk_size:(i + 1) * chunk_size])
        select_query = query(selected_samples)
        conn_db.execute(select_query)
        rows = conn_db.fetchall()
        print "num_chunk", i
        for row in rows:
            csku_object = (pickle.loads(str(row[0])))
            csku_id = csku_object['Config SKU']
            dict_cgs[csku_id] = {COMMODITY_GROUPS: csku_object[COMMODITY_GROUPS],
                                 CATEGORY: csku_object[CATEGORY],
                                 SEASON_TYPE: csku_object[SEASON_TYPE]}
    with open(output_file, 'wb') as dict_cgs_file:
        pickle.dump(dict_cgs, dict_cgs_file)


compute_cg_dict(training_db_file, 'dict_cgs.pkl')
