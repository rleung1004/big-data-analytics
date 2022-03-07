import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def process_data():
    data_confirmed = pd.read_csv("./covid_19_clean_complete_2022.csv")

    # Look at data structure
    print(data_confirmed.head())

    # Get confirmed cases from Canada
    df_canada = data_confirmed[data_confirmed['Country/Region'] == 'Canada']
    df_canada['Date'] = pd.to_datetime(df_canada['Date'])
    df_canada = df_canada.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

    print(df_canada)
    print(df_canada.describe())

    df_canada = df_canada.drop(['Deaths', 'Recovered', 'Active'], axis=1)

    return df_canada


def create_time_shift_columns(df, col_name, n):
    df = df.copy(True)
    for i in range(1, n + 1):
        new_col_name = f"{col_name}_t-{i}"
        df[new_col_name] = df[col_name].shift(periods=i)

    return df


def plot_dataset(df, col_name, title):
    df[col_name].plot()
    plt.title(title)
    plt.show()


def t_series_decomposition(df, col_name):
    t_series = seasonal_decompose(df[col_name], model='additive', extrapolate_trend='freq')
    t_series.plot()
    plt.show()


def auto_correlation():
    df["d1"] = diff(df['Confirmed'], k_diff=1)
    df["d2"] = diff(df['Confirmed'], k_diff=2)

    plot_acf(df[1:].d1.values.squeeze(), lags=40)
    plot_pacf(df[1:].d1.values.squeeze(), lags=40)
    plt.show()

    plot_acf(df[2:].d2.values.squeeze(), lags=40)
    plot_pacf(df[2:].d2.values.squeeze(), lags=40)
    plt.show()

    t_series = seasonal_decompose(df['Confirmed'], model="additive")
    t_series.plot()
    plt.show()


col = "Confirmed"
df = process_data()
df = create_time_shift_columns(df, 'Confirmed', 10)
plot_dataset(df, col, "COVID-19 Cases Total Per Day from Jan 2020 to Mar 2022")
auto_correlation()
