from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

NUM_TEST_DAYS = 30
TEST_VAL_SPLIT = 15


def get_processed_data():
    data_confirmed = pd.read_csv("./covid_19_clean_complete_2022.csv")
    # Get confirmed cases from Canada
    df_canada = data_confirmed[data_confirmed['Country/Region'] == 'Canada']
    df_canada['Date'] = pd.to_datetime(df_canada['Date'])
    df_canada = df_canada.groupby(['Date'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()

    df_canada = df_canada.drop(['Deaths', 'Recovered', 'Active'], axis=1)
    df_canada.reset_index(inplace=True)
    print(df_canada)
    return df_canada


def create_time_shift_columns(df, col_name, n):
    df = df.copy(True)
    for i in range(1, n + 1):
        new_col_name = f"{col_name}_t-{i}"
        df[new_col_name] = df[col_name].shift(periods=i)

    return df


df = get_processed_data()
df = create_time_shift_columns(df, 'Confirmed', 100)
df = df.dropna()

# Train test split data
y = df['Confirmed']
x = df.drop(['Confirmed', 'Date'], axis=1)

train_test_partition = len(df) - NUM_TEST_DAYS
x_train = x[0:train_test_partition]
y_train = y[0:train_test_partition]
x_test = x[train_test_partition:train_test_partition + TEST_VAL_SPLIT]
y_test = y[train_test_partition:train_test_partition + TEST_VAL_SPLIT]
x_val = x[train_test_partition + TEST_VAL_SPLIT:]
y_val = y[train_test_partition + TEST_VAL_SPLIT:]

x_best = SelectKBest(score_func=f_regression, k=100)
x_best.fit(x_train, y_train)

mask = x_best.get_support()
new_feat = []

for selected, feature in zip(mask, x_train.columns):
    if selected:
        new_feat.append(feature)
print('The best features are {}'.format(new_feat))
