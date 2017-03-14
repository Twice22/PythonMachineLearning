from sklearn.svm import SVC

from plottingDecisionRegions import * # homemade file
from xorgate import * # homemade file

# we use the rbf kernel (Gaussian kernel) see p77
# gamma = 1/(2*sigma²)
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# The γ parameter, which we set to gamma=0.1, can be understand as
# a cut-off param for the Gaussian sphere. if γ increases, we
# increase the influence of the training samples :
# softer decision boundary.