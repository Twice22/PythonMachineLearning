from perceptron import Perceptron
from readingIris import X, y
import matplotlib.pyplot as plt

# training the Perceptron algorithm
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

# following lines won't be executed if this file is
# imported from another Python file
if __name__ == "__main__":
	plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Number of misclassifications')
	plt.show()