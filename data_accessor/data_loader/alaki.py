import sqlite3
import pandas as pd
import datetime
from datetime import datetime, timedelta
import isoweek
from isoweek import Week


def get_compute_relative_forecast_week_function(start_date_str):
    start_date = Week.fromstring(start_date_str)

    def compute_relative_forecast_week(df_row):
        current_year = (df_row['year'])
        current_week = int(df_row['week'])
        current_date = Week(current_year, current_week)

        return current_date - start_date

    return compute_relative_forecast_week


db_path = "/Users/tnaghibi/backtesting_data/accuracy_metrics_201545.db"
conn = sqlite3.connect(db_path)
con_curs = conn.cursor()
sibr_obs = "sibr_obs"
sibr_fc = "sibr_fc"
black_price = "black_price"
risk_date = 'risk_discounting_start_date'
end_season = 'season_end_date'
csku_id = 'csku'
query = "Select * from BT_ACCURACY_METRICS where week=45"

con_curs.execute(query)
a_raw = pd.read_sql_query(query, conn)
a_raw['start_risk'] = a_raw[[risk_date]].apply(lambda x: datetime.strptime(x[0], '%Y-%m-%d').date(), axis=1)
a_raw['end_season'] = a_raw[[end_season]].apply(lambda x: datetime.strptime(x[0], '%Y-%m-%d').date(), axis=1)
compute_relative_forecast_week_function = get_compute_relative_forecast_week_function('2015W45')
# relative_forecast_week = a_raw[['year', 'week']].apply(compute_relative_forecast_week_function, axis=1)
weights = a_raw['start_risk'] <= isoweek.Week.fromstring("2015W45").monday()

a_raw['week_year'] = a_raw[['week', 'year']].apply(
    lambda x: isoweek.Week.fromstring(str(x[1]) + 'W' + "{0:0=2d}".format(x[0])).monday(), axis=1)
weights &= a_raw['week_year'] <= a_raw['end_season']

import numpy as np

# week_decay = relative_forecast_week.apply(lambda week_index: np.exp(np.multiply(week_index, -0.1)))
weights = weights
print len(a_raw)
# a_raw = a_raw[weights]
print len(a_raw)
aggregations = {sibr_obs: ['sum'], sibr_fc: ['sum'], black_price: ['mean'], 'weight': ['mean']}
a_raw['weight'] = 1
# a=a_raw
a = a_raw.groupby([csku_id, 'week', 'year']).agg(aggregations).reset_index()
a.columns = a.columns.droplevel(1)
print a.columns
print weights[0:20]
b = a[black_price] * a['weight']
print (b[0:20], weights[0:20])
kpi = 1 - sum(abs(a[sibr_obs] - a[sibr_fc]) * b) / sum(a[sibr_obs] * b)
print kpi
