def example_1():
    import pandas_datareader as pdr
    import matplotlib.pyplot as plt
    import datetime

    # Retrieve stock data.
    def getStock(stockName):
        start = datetime.datetime(2019, 6, 1)
        end = datetime.date.today()
        stk = pdr.get_data_yahoo(stockName,
                                 start=datetime.datetime(start.year,
                                                         start.month, start.day),
                                 end=datetime.datetime(end.year, end.month, end.day))
        return stk

    stockJPM = getStock('JPM')
    stockZoom = getStock('ZM')

    # Plot closing stock prices.
    plt.plot(stockJPM.index, stockJPM.Close, label='JPM', color='blue')
    plt.plot(stockZoom.index, stockZoom.Close, label='Zoom', color='red')
    plt.title("Stock Closing Prices", fontsize=20)
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

    # Plot daily % returns.
    JPMReturns = 100 * stockJPM.Close.pct_change().dropna()
    zoomReturns = 100 * stockZoom.Close.pct_change().dropna()
    plt.plot(JPMReturns.index, JPMReturns, label='JPM', color='blue')
    plt.plot(zoomReturns.index, zoomReturns, label='Zoom', color='red', alpha=0.3)
    plt.ylabel('Percent Return')
    plt.title('Daily Percent Returns', fontsize=20)
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(JPMReturns ** 2, lags=50)
    plt.show()

    plot_pacf(zoomReturns ** 2, lags=50)
    plt.show()

    from arch import arch_model
    import pandas as pd

    def gridSearchArchModel(pRange, returns):
        df = pd.DataFrame(columns=['p', 'aic', 'bic'])

        for p in range(1, pRange):
            model = arch_model(returns, p=p, q=0)
            model_fit = model.fit()
            df = df.append({'aic': model_fit.aic, 'bic': model_fit.bic, 'p': p},
                           ignore_index=True)
        print("ARCH Model Results")
        print(df)

    # MAX_P = 14
    # gridSearchArchModel(MAX_P, zoomReturns)

    JPM_LAGS = 5
    model = arch_model(JPMReturns, p=JPM_LAGS, q=0)
    model_fit = model.fit()
    print(model_fit.summary())

    TEST_DAYS = 170
    ZOOM_LAGS = 3
    model = arch_model(zoomReturns, p=ZOOM_LAGS, q=0)
    model_fit = model.fit()
    print(model_fit.summary())
    import numpy as np

    def generatePredictions(returns, numLags):
        dayAheadPredictions = []
        for i in range(TEST_DAYS):
            train = returns[:-(TEST_DAYS - i)]

            # This code is just proofing out the fact that the forecast is for
            # 1 day ahead. The sample output generated is:
            # Most recent date in training data: 2020-10-26 00:00:00
            # Prediction is for date: 2020-10-27 00:00:00
            lastTrainingDate = returns.index[len(train) - 1]
            print("\nMost recent date in training data: " + str(lastTrainingDate))
            print("Prediction is for date: " + str(returns.index[len(train)]))

            # Fit the model, make the day-ahead prediction, add prediction to the list.
            model = arch_model(train, p=numLags, q=0)
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)  # Predict 1 day ahead
            dayAheadPredictions.append(pred.variance.values[-1, :][0])
        return dayAheadPredictions

    zoomDayAheadPredictions = generatePredictions(zoomReturns, ZOOM_LAGS)

    def plotPredictedVolatility(returns, dayAheadPredictions, stockName):
        actualRecentData = returns[len(returns) - TEST_DAYS:]

        plt.plot(actualRecentData.index, actualRecentData, marker="o",
                 label="Actual % Price Change")
        plt.plot(actualRecentData.index, dayAheadPredictions, marker="o",
                 label="Predicted % Deviation")
        plt.axhline(y=0, color='gray', alpha=0.3)
        plt.legend()
        plt.xticks(rotation=60)
        plt.title("Predicted Daily σ (sigma) for " + stockName)
        plt.show()

    plotPredictedVolatility(zoomReturns, np.sqrt(zoomDayAheadPredictions), "Zoom")

    import pandas as pd
    def multiDayForecast(returns, pLags, color, stockName):
        train = returns
        model = arch_model(train, p=pLags, q=0)
        model_fit = model.fit(disp='off')

        NUM_DAYS_AHEAD = 7
        pred = model_fit.forecast(horizon=NUM_DAYS_AHEAD)
        future_dates = [returns.index[-1] + datetime.timedelta(days=i) for i in range(1, NUM_DAYS_AHEAD + 1)]
        pred = pd.Series(pred.variance.values[-1, :], index=future_dates)
        plt.plot([1, 2, 3, 4, 5, 6, 7], np.sqrt(pred), color=color, marker='o', label=stockName)

    plt.figure(figsize=(10, 4))
    multiDayForecast(JPMReturns, JPM_LAGS, 'blue', 'JP Morgan')
    ZOOM_LAGS = 1
    multiDayForecast(zoomReturns, ZOOM_LAGS, 'red', 'Zoom')
    plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
    plt.axhline(y=0, color='gray', alpha=0.3)
    plt.legend(loc='best')
    plt.show()


def example_2():
    import pandas_datareader as pdr
    import matplotlib.pyplot as plt
    import datetime
    from arch import arch_model
    import pandas as pd

    # Retreive stock data.
    def getStock(stockName):
        start = datetime.datetime(2019, 6, 1)
        end = datetime.date.today()
        stk = pdr.get_data_yahoo(stockName,
                                 start=datetime.datetime(start.year,
                                                         start.month, start.day),
                                 end=datetime.datetime(end.year, end.month, end.day))
        return stk

    stockJPM = getStock('JPM')
    stockZoom = getStock('ZM')

    # Plot daily % returns.
    JPMReturns = 100 * stockJPM.Close.pct_change().dropna()
    zoomReturns = 100 * stockZoom.Close.pct_change().dropna()

    def gridSearchGarchModel(returns):

        df = pd.DataFrame(columns=['p', 'q', 'aic', 'bic'])

        MAX_LAGS = 10
        for p in range(1, MAX_LAGS):
            for q in range(1, MAX_LAGS):
                model = arch_model(returns, p=p, q=q)
                model_fit = model.fit()
                df = df.append({'aic': model_fit.aic,
                                'bic': model_fit.bic, 'p': p, 'q': q},
                               ignore_index=True)
        df.sort_values("bic")
        print("ARCH Model Results")
        print(df)

    gridSearchGarchModel(JPMReturns)

    model = arch_model(JPMReturns, p=1, q=1)
    model_fit = model.fit()
    print(model_fit.summary())


example_2()
