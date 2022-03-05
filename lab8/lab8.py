from statsmodels.stats.power import TTestIndPower

effect = 2.36  # Obtained from previous step.
alpha = 0.05  # Enable 95% confidence for two tail test.
power = 0.95  # One minus the probability of a type II error.
# Limits possibility of type II error to 20%.
analysis = TTestIndPower()
numSamplesNeeded = analysis.solve_power(effect, power=power, alpha=alpha)
print(numSamplesNeeded)

from scipy import stats
before_shut_down = [10, 11, 6, 18, 11, 9]
after_shut_down = [5, 7, 3, 12, 0, 7]

testResult = stats.ttest_ind(before_shut_down,
                             after_shut_down, equal_var=False)

import numpy as np
print("Hypothesis test p-value: " + str(testResult))
print("New sales mean: " + str( np.mean(after_shut_down)))
print("New sales std: " + str(np.std(after_shut_down)))

