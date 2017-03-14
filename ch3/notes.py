# we can initialize the stochastic gradient descent version
# of perceptron, logistic reg and svm with default param as
# follows :
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='percetron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')