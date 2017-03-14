# we gonna use the MNIST dataset

# Training set images : 60 000 samples
# Training set labels : 60 000 labels
# Test set images :  	10 000 samples
# Test set labels :		10 000 labels

# image are stored in byte format and we'll use Numpy arrays to read them
import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
	"""Load MNIST data from path"""
	labels_path = os.path.join(path,
							   '%s-labels.idx1-ubyte' % kind)

	images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

	with open(labels_path, 'rb') as lbpath:
		# read 8 first bits of the MNIST dataset (magic number and nb
		# of items : infos that we don't want to get)
		# > is for big-endian (defines the order in which a seq
		# is stored)
		# I : This is an unsigned integer
		magic, n = struct.unpack('>II', lbpath.read(8))

		# read binary data
		# dtype : Data type of the returned array. Used
		# to determine the size and byte-order of the items in
		# the file.
		labels = np.fromfile(lbpath, dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))

		images = np.fromfile(imgpath, dtype=np.uint8).reshape(
											   len(labels), 784)

	return images, labels

X_train, y_train = load_mnist('mnist', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))


from neuralnet import NeuralNetMLP

# new neural network with 10 hidden layers, no regularization (l1, l2) = (0, 0)
# no adaotuve learning, no momentum learning and regular gradient descent
# using minibatches to 1
nn_check = NeuralNetMLP(n_output=10,
						n_features=X_train.shape[1],
						n_hidden=10,
						l2=0.0,
						l1=0.0,
						epochs=10,
						eta=0.001,
						alpha=0.0,
						decrease_const=0.0,
						minibatches=1,
						random_state=1)

nn_check.fit(X_train[:5], y_train[:5], print_progress=False)