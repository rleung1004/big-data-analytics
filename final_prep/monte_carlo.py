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
