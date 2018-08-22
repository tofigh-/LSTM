import sqlite3
db_path = "/Users/tnaghibi/Downloads/accuracy_metrics_201645.db"
conn = sqlite3.connect(db_path)
con_curs = conn.cursor()
sibr_obs = "sibr_obs"
sibr_fc = "sibr_fc"
black_price = "black_price"
risk_date = 'risk_discounting_start_date'
end_season = 'season_end_date'
csku_id = 'csku'
query = "Select sum(abs(sibr_fc-sibr_obs)*black_price) / sum(sibr_obs * black_price) from BT_ACCURACY_METRICS where week=45 and country=0agit "
con_curs.execute(query)
print con_curs.fetchall()