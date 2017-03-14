# from adaline import *
from adalineStochastic import *
from readingIris import *
from plottingDecisionRegions import plot_decision_regions

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# using eta = 0.01, will see that Adaline now converges
# ada = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y) # homemade function
# plot_decision_regions(X_std, y, classifier=ada) # homemade function

# Adaline for stochastic gradient descent
ada = AdalineGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y) # homemade function
plot_decision_regions(X_std, y, classifier=ada) # homemade function


# plt.title('Adaline - Gradient Descent')
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
plt.ylabel('Average Cost')
plt.show()