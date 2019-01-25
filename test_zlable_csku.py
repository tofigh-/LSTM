import pandas as pd

data = pd.read_csv("/Users/tnaghibi/Downloads/zlabels_skus",delimiter=";")
print data.Date.unique()