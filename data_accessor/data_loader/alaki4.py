import sqlite3
import cPickle as pickle
from Settings import *
from utilities import get_supplier_stock_uplift

training_db_file = "/Users/tnaghibi/PycharmProjects/data_accessor/model/training.db"
live_db_file = "/Users/tnaghibi/backtesting_data/live_data/data3_201702_production.db"


def get_csku(db_file):
    connection = sqlite3.connect(db_file)
    conn_db = connection.cursor()
    csku_id = "'0AE81A000-A11'"
    query = 'SELECT dictionary FROM data WHERE csku = {csku_id}'.format(csku_id=csku_id)
    conn_db.execute(query)
    rows = conn_db.fetchall()
    for row in rows:
        csku_object1 = (pickle.loads(str(row[0])))
        print csku_object1['Config SKU']
    return csku_object1
csku1 = get_csku(training_db_file)
csku2 = get_csku(live_db_file)

s1= csku1[SALES_MATRIX][:,(106-52):107]
s2= csku2[SALES_MATRIX][:,(106-52):107]

print csku1[BLACK_PRICE_INT] - csku2[BLACK_PRICE_INT]
print csku1[CATEGORY],"kir ",csku2[CATEGORY]
print csku1[STOCK][54:107] - csku2[STOCK][54:107]
csku2[STOCK_UPLIFT] = get_supplier_stock_uplift(csku2[STOCK],
                                          np.sum(csku2[SALES_MATRIX], 0),
                                          np.sum(csku2[RETURNS_TIMES_MATRIX], 0))
print csku1[STOCK_UPLIFT][54:107] - csku2[STOCK_UPLIFT][54:107]