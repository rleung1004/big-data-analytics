import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import pmdarima as pm
from statsmodels.tsa.statespace.tools import diff


def get_processed_data():
    data_confirmed = pd.read_csv("./covid_19_clean_complete_2022.csv")
    # Get confirmed cases from Canada
    df_canada = data_confirmed[data_confirmed['Country/Region'] == 'Canada']
    df_canada['Date'] = pd.to_datetime(df_canada['Date'])
    df_canada = df_canada.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

    df_canada = df_canada.drop(['Deaths', 'Recovered', 'Active'], axis=1)
    df_canada.reset_index(inplace=True)
    df_canada["d1"] = diff(df_canada['Confirmed'], k_diff=1)
    df_canada["d2"] = diff(df_canada['Confirmed'], k_diff=2)
    return df_canada


def create_time_shift_columns(df, col_name, n):
    df = df.copy(True)
    for i in range(1, n + 1):
        new_col_name = f"{col_name}_t-{i}"
        df[new_col_name] = df[col_name].shift(periods=i)

    return df


def evaluate_model(y_test, predictions, model):
    MSE = metrics.mean_squared_error(y_test, predictions)
    MAE = metrics.mean_absolute_error(y_test, predictions)
    RMSE = metrics.mean_squared_error(y_test, predictions, squared=False)
    MAPE = metrics.mean_absolute_percentage_error(y_test, predictions)
    print('-------------------- Error Data Testing -----------------')
    print(model.__class__.__name__)
    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("RMSE: ", RMSE)
    print("MAPE: ", MAPE)


def ols_score_func(t_3, t_4, t_9):
    return 1.8944 * t_3 - 0.5119 * t_4 - 0.3824 * t_9


def main():
    NUM_TEST_DAYS = 30
    TEST_VAL_SPLIT = 15
    # Create a dataframe to store our predictions from each model
    df_predictions = pd.DataFrame()

    df = get_processed_data()
    df = create_time_shift_columns(df, 'Confirmed', 10)
    df = df.dropna()

    features = ['Confirmed_t-3', 'Confirmed_t-4', 'Confirmed_t-9']
    validation_features = ['Confirmed_t-2', 'Confirmed_t-3', 'Confirmed_t-8']
    x = sm.add_constant(df)
    # Train test split data
    y = df['Confirmed']
    x = df[features]
    t_validation = df[validation_features]

    train_test_partition = len(df) - NUM_TEST_DAYS
    x_train = x[0:train_test_partition]
    y_train = y[0:train_test_partition]
    x_test = x[train_test_partition:train_test_partition + TEST_VAL_SPLIT]
    y_test = y[train_test_partition:train_test_partition + TEST_VAL_SPLIT]
    x_val = x[train_test_partition + TEST_VAL_SPLIT:]
    y_val = y[train_test_partition + TEST_VAL_SPLIT:]
    t_validation = x[train_test_partition + TEST_VAL_SPLIT:]

    # Scale data
    x_scaler, y_scaler = RobustScaler(), RobustScaler()
    x_scaled = x_scaler.fit_transform(x_train)
    y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    x_test_scaled = x_scaler.transform(x_test)

    ols_model = sm.OLS(y_scaled, x_scaled).fit()
    print(ols_model.summary())

    ols_predictions = ols_model.predict(x_test_scaled)
    df_predictions['OLS'] = ols_predictions
    arima = pm.auto_arima(y_train, start_p=0, start_q=0, max_p=7, max_q=7, seasonal=False, test='adf',
                          d=0, max_d=2, trace=True, enforce_stationarity=False, enforce_invertibility=False,
                          error_action='ignore', suppress_warnings=True, maxiter=50, stepwise=True)

    print(arima.summary())
    # Forecast
    n_periods = 15
    arima_pred = arima.predict(n_periods=n_periods)
    df_predictions['ARIMA'] = arima_pred

    # Create dataframe for validation data
    df_validation = pd.DataFrame()
    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")

    scaled_validation = ols_model.predict(x_scaler.transform(x_val))
    df_validation['OLS'] = scaled_validation
    ols_validation = y_scaler.inverse_transform(scaled_validation.reshape(-1, 1))
    evaluate_model(y_val, ols_validation, ols_model)

    arima_validation = arima.predict(n_periods=15)
    df_validation['ARIMA'] = arima_validation
    evaluate_model(y_val, arima_validation, arima)

    stacked_model = LinearRegression()
    stacked_model.fit(df_predictions, y_test)
    stacked_pred = stacked_model.predict(df_validation)

    print("\n** Evaluate Stacked Model **")
    evaluate_model(y_val, stacked_pred, stacked_model)

    df_forecast = pd.DataFrame()
    # Get last entry of dataframe, this is t - 0
    scaled_val = x_scaler.transform(t_validation)
    t = scaled_val[len(scaled_val) - 1]

    df_forecast['OLS'] = [ols_score_func(t[0], t[1], t[2])]
    df_forecast['ARIMA'] = arima.predict(n_periods=1)

    t_plus_1 = stacked_model.predict(df_forecast)
    print(t_plus_1)


main()
