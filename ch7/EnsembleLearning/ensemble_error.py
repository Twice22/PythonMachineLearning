# me assumes that :
		# - we have n base classifier
		# - they all have an error rate of epsilon
		# - classifiers are independant, error rates not correlated

# so the error of the ensemble is a binomial distribution !
from scipy.misc import comb
import math

def ensemble_error(n_classifier, error):
	# half of the classifier made an error in this example :
	k_start = math.ceil(n_classifier / 2.0)
	# comb : n choose k (k parmi n)
	# range go from k_start included to (n_classifier + 1) excluded
	probs = [comb(n_classifier, k) * error**k *
			(1-error)**(n_classifier - k)
			for k in range(k_start, n_classifier +1)]
	return sum(probs)

print(ensemble_error(n_classifier=11, error=0.25))


# plot the relationship between ensemble and base errors in a
# line graph :

import numpy as np
# array from 0.0 to 1.01 every 0.01
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
			 for error in error_range]

import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors,
		 label='Ensemble error',
		 linewidth=2)
plt.plot(error_range, error_range,
		label='Base error',
		linestyle='--')
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# the ensemble error performs better than the individual base
# classifier as long as the base classifiers perfom better
# than random guessing