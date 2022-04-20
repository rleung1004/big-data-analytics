import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pandas_datareader  as pdr

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Do not show warning.
pd.options.mode.chained_assignment = None  # default='warn'

##################################################################
# CONFIGURATION SECTION
NUM_TIME_STEPS  = 3
TEST_DAYS       = 7
##################################################################
PATH = "../datasets/"
FILE = "mytime_series.csv"
df = pd.read_csv(PATH + FILE)


def create_time_shift_columns(df, col_name, n):
    df = df.copy(True)
    for i in range(1, n + 1):
        new_col_name = f"{col_name}_t-{i}"
        df[new_col_name] = df[col_name].shift(periods=i)

    return df

df = create_time_shift_columns(df, "target", 3)
df = create_time_shift_columns(df, "a", 3)
df = df.dropna()

X = df[['a_t-1', 'target_t-1', 'target_t-3']]
y = df['target']

print(df.tail())

# Add intercept for OLS regression.
import statsmodels.api       as sm
X = sm.add_constant(X)

# Split into test and train sets. The test data must be
# the latest data range.
lenData = len(X)
X_train = X[0:lenData-TEST_DAYS]
y_train = y[0:lenData-TEST_DAYS]
X_test  = X[lenData-TEST_DAYS:]
y_test  = y[lenData-TEST_DAYS:]

# Model and make predictions.
model       = sm.OLS(y_train, X_train).fit()
print(model.summary())
predictions = model.predict(X_test)

# Show RMSE and plot the data.
from sklearn  import metrics
import numpy as np
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print("One day ahead prediction")
print(1.5547*-21 + 1.0844*12 - 0.1785*-22 + 30.3605)
