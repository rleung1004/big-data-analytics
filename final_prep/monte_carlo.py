NUM_SIMULATIONS = 100

# Generate random numbers from a normal distribution.
from scipy.stats import norm
def generateRandomNumbers(mean, standardDev):
    randomNums = norm.rvs(loc=mean,
                          scale=standardDev,
                          size=NUM_SIMULATIONS)
    return randomNums

revenue = generateRandomNumbers(234000, 18000)
rental = generateRandomNumbers(65000, 0)
labour = generateRandomNumbers(50000, 20000)
expense = generateRandomNumbers(110000, 15000)

import pandas as pd
df = pd.DataFrame(columns=['Revenue', 'Rental', 'Labour', 'Expenses'])

for i in range(0, NUM_SIMULATIONS):
    dictionary = {
        'Revenue': round(revenue[i]),
        'Rental': round(rental[i]),
        'Labour': round(labour[i]),
        'Expenses': round(expense[i]),
        'Net': round(revenue[i]) - round(rental[i]) - round(labour[i]) - round(expense[i])
    }
    df = df.append(dictionary, ignore_index=True)

# Show the data frame which contains results from
# all 100 trials.
print(df)

# Calculate profit summaries.
print("Net Mean: " + str(df['Net'].mean()))
print("Net SD:   " + str(df['Net'].std()))
print("Net Min:  " + str(df['Net'].min()))
print("Net Max:  " + str(df['Net'].max()))

# Calculate the risk of incurring a loss.
dfLoss = df[(df['Net'] < 0)]
totalLosses = dfLoss['Net'].count()
riskOfLoss = totalLosses / NUM_SIMULATIONS
print("Risk of loss: " + str(riskOfLoss))
