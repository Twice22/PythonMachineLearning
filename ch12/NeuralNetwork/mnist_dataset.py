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

X_test, y_test = load_mnist('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


# let's see sample of digits 0-9 after reshaping the 784-pixel vectors
# into the 28x28 original image

import matplotlib.pyplot as plt
# fig with 2 row, 5 cols and X are shared between image of same col
# Y are shared between image of same row
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

# return copy of the array collapsed into one dimension
# if we don't use flatten() we need to do (in the loop)
# ax[i][j].imshow (for i in range(5) and j in range (2))
ax = ax.flatten()
for i in range(10):
	# reshape 784 long vector into a 28x28 matrix
	img = X_train[y_train == i][0].reshape(28, 28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')

# remove the x and y ticks (graduation)
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# let's plot multiple examples of the same digit to see the difference
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
	img = X_train[y_train==2][i].reshape(28,28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# remove the following comments to save those datasets to csv files
# note : It will take a lot of place !
# fmt allow us to specfy the format
# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')

# to load them back we can do :
# X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')


# let's initialize a neural network with 784 inputs units (n_features)
# 50 hidden units (n_hidden) and 10 output units (n_output):
from neuralnet import NeuralNetMLP
nn = NeuralNetMLP(n_output=10,
				  n_features=X_train.shape[1],
				  n_hidden=50,
				  l2=0.1,
				  l1=0.0,
				  epochs=1000,
				  eta=0.001,
				  alpha=0.001,
				  decrease_const=0.00001,
				  shuffle=True,
				  minibatches=50,
				  random_state=1)

# l2 : lambda param for L2 reg to decrease the degree of overfitting
# l1 : same for L1 regularization
# epochs: nb of passes over the training set.
# eta : learning rate
# alpha : A param for momentum learning to add a factor of the previous
#		  to the weight update for faster learning.
# decrease_const : to decrease learning reate eta for better convergence
# shuffle : Shuffling the training set prior to every epoch to prevent
#			the algo from getting stuck in cycles.
# Minibatches: Splitting of the training data into k mini-batches in
#			each epoch. gradient computed for each mini-batch -> faster
#			learning


nn.fit(X_train, y_train, print_progress=True)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

# let's plot de cost function again by averaging over
# the mini-batch

# mean over 1000 cost each time
batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)),
		 cost_avgs,
		 color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# evaluate the performance of the model (prediction accuracy):
y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

# accuracy on 10 000 imags in the dataset
y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))


# let's take a look at some of the images that our MLP struggles with
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten() # use to avoid doing ax[i][j].imshow in for loop
for i in range(25):
	img = miscl_img[i].reshape(28, 28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')
	ax[i].set_title('%d) t: %d p: %d' 
					% (i+1, correct_lab[i], miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
