def monte_carlo_simulation_example():
    NUM_SIMULATIONS = 500

    # Define revenue, fixed and variable cost parameters.
    REV_EXPECTED = 194000
    REV_SD       = 15000

    FC_EXPECTED  = 60000
    FC_SD        = 4000

    VC_EXPECTED  = 100000
    VC_SD        = 40000

    # Generate random numbers from a normal distribution.
    from scipy.stats import norm
    def generateRandomNumbers(mean, standardDev):
        randomNums = norm.rvs(loc   = mean,
                              scale = standardDev,
                              size  = NUM_SIMULATIONS)
        return randomNums

    revenues      = generateRandomNumbers(REV_EXPECTED, REV_SD)
    fixedCosts    = generateRandomNumbers(FC_EXPECTED, FC_SD)
    variableCosts = generateRandomNumbers(VC_EXPECTED, VC_SD)

    profits = []

    import pandas as pd
    df = pd.DataFrame(columns = ['Revenue', 'Fixed Cost',
                                 'Variable Cost', 'Profit'])

    for i in range(0, NUM_SIMULATIONS):
        profit     = revenues[i] - fixedCosts[i] - variableCosts[i]

        dictionary = {'Revenue':      round(revenues[i],2),
                      'Fixed Cost':   round(fixedCosts[i],2),
                      'Variable Cost':round(variableCosts[i],2),
                      'Profit':       round(profit,2)}
        df = df.append(dictionary, ignore_index=True)

    # Show the data frame which contains results from
    # all 500 trials.
    print(df)

    # Calculate profit summaries.
    print("Profit Mean: " + str(df['Profit'].mean()))
    print("Profit SD:   " + str(df['Profit'].std()))
    print("Profit Min:  " + str(df['Profit'].min()))
    print("Profit Max:  " + str(df['Profit'].max()))

    # Calculate the risk of incurring a loss.
    dfLoss      = df[(df['Profit']<0)]
    totalLosses = dfLoss['Profit'].count()
    riskOfLoss  = totalLosses/NUM_SIMULATIONS
    print("Risk of loss: " + str(riskOfLoss))


def bank_machine_simulation():
    NUM_SIMULATIONS = 500

    # Define revenue, fixed and variable cost parameters.
    CUSTOMERS_EXPECTED = 23
    CUSTOMERS_SD = 7

    # Generate random numbers from a normal distribution.
    from scipy.stats import norm
    def generateRandomNumbers(mean, standardDev):
        randomNums = norm.rvs(loc=mean,
                              scale=standardDev,
                              size=NUM_SIMULATIONS)
        return randomNums

    customers = generateRandomNumbers(CUSTOMERS_EXPECTED, CUSTOMERS_SD)

    import pandas as pd
    df = pd.DataFrame(columns=['Customers'])

    for i in range(0, NUM_SIMULATIONS):
        dictionary = {'Customers': round(customers[i])}
        df = df.append(dictionary, ignore_index=True)

    # Show the data frame which contains results from
    # all 500 trials.
    print(df)

    # Calculate profit summaries.
    print("Customers Mean: " + str(df['Customers'].mean()))
    print("Customers SD:   " + str(df['Customers'].std()))
    print("Customers Min:  " + str(df['Customers'].min()))
    print("Customers Max:  " + str(df['Customers'].max()))

    # Calculate the risk of incurring a loss.
    dfLoss = df[(df['Customers'] > 30)]
    totalLosses = dfLoss['Customers'].count()
    riskOfLoss = totalLosses / NUM_SIMULATIONS
    print("Risk of loss: " + str(riskOfLoss))


def uniform_number_generation():
    import numpy as np
    import matplotlib.pyplot as plt
    LOW = 1
    HIGH = 24
    SIZE = 100
    NUM_SIMULATIONS = 5

    plt.subplots(nrows=5, ncols=1, figsize=(14, 7))
    num_bankrupts = []
    for i in range(1, NUM_SIMULATIONS + 1):
        # Randomize data
        x = np.random.uniform(LOW, HIGH, SIZE)
        plt.subplot(2, 3, i)
        plt.hist(x, 24, density=True)
        num_bankrupts.append(len(np.where(x < 2)[0]))
    plt.show()
    bankrupt_rate = np.mean(num_bankrupts) / SIZE
    print("Bankrupt Rate: " + str(bankrupt_rate))


def drug_sales():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats
    from scipy.stats import kstest

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Load samples.
    PATH = '../datasets/'
    FILE = 'drugSales.csv'
    df = pd.read_csv(PATH + FILE)
    samples = np.array(df[['value']])

    # High p-values are preferred and low D scores (closer to 0)
    # are preferred.
    def runKolmogorovSmirnovTest(dist, loc, arg,
                                 scale, samples):
        d, pvalue = kstest(samples.tolist(),
                           lambda x: dist.cdf(x,
                                              loc=loc, scale=scale, *arg),
                           alternative="two-sided")
        print("D value: " + str(d))
        print("p value: " + str(pvalue))
        dict = {"dist": dist.name, "D value": d, "p value": pvalue,
                "loc": loc, "scale": scale, "arg": arg}
        return dict

    def fit_and_plot(dist, samples, df):
        print("\n*** " + dist.name + " ***")

        # Fit the distribution as best as possible
        # to existing data.
        params = dist.fit(samples)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Generate 'x' values between 0 and 80.
        x = np.linspace(0, 80, 80)

        # Run test to see if generated data aligns properly
        # to the sample data.
        distDandP = runKolmogorovSmirnovTest(dist, loc, arg, scale, samples)
        df = df.append(distDandP, ignore_index=True)

        # Plot the test and actual values together.
        _, ax = plt.subplots(1, 1)
        plt.hist(samples, bins=80, range=(0, 80))
        ax2 = ax.twinx()
        ax2.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg),
                 '-', color="r", lw=2)
        plt.title(dist.name)
        plt.show()
        return df

    distributions = [
        scipy.stats.norm,
        scipy.stats.gamma,
        scipy.stats.chi2,
        scipy.stats.wald,
        scipy.stats.uniform,
        scipy.stats.norm,
        scipy.stats.t,
        scipy.stats.chi,
        scipy.stats.dgamma,
        scipy.stats.ncx2,
        scipy.stats.burr,
        scipy.stats.bradford,
        scipy.stats.crystalball,
        scipy.stats.exponnorm,
        scipy.stats.pearson3,
        scipy.stats.exponpow,
        scipy.stats.invgauss,
        scipy.stats.argus,
        scipy.stats.gennorm,
        scipy.stats.expon,
        scipy.stats.gengamma,
        scipy.stats.genextreme,
        scipy.stats.genexpon,
        scipy.stats.genlogistic]

    dfDistribution = pd.DataFrame()
    # Grid search the continuous distributions.
    for i in range(0, len(distributions)):
        dfDistribution = fit_and_plot(distributions[i], samples, dfDistribution)

    dfDistribution = dfDistribution.sort_values(by=['D value'])
    print(dfDistribution.T)  #


drug_sales()
