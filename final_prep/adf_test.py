import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
PATH = "../datasets/"
FILE = 'HydroConsumption.csv'
df   = pd.read_csv(PATH + FILE)

df["hydro"].plot()
plt.title("Hydro Consumption")
plt.show()

from statsmodels.tsa.stattools import adfuller
result = adfuller(df.hydro.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Perform differencing.
df = df.diff()

# Plot data after differencing.
plt.plot(df)
plt.xticks(rotation=75)
plt.show()

result = adfuller(df.hydro.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Perform differencing.
df = df.diff()

# Plot data after differencing.
plt.plot(df)
plt.xticks(rotation=75)
plt.show()

result = adfuller(df.hydro.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
