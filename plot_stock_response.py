import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/Users/tnaghibi/PycharmProjects/data_accessor/model/stock_Response",header=None)
d_sum = data.mean().values
plt.figure()
plt.plot(data[0:10].values.transpose())
# plt.plot(np.arange(51) / 10.0, d_sum)
plt.show()
