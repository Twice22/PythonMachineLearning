from adaline import *
from readingIris import *

# number of rows of the subplot grid
# number of col of the subplot grid
# tuple of integers with width and height in inches
# return :
#	- a tuple containing figure and axes object(s)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# first fig : eta = 0.01
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# second fig : eta = 0.0001
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')


plt.show()