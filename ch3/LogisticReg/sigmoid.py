import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
	return 1.0 / ( 1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1) #array from -7 to 7 every 0.1
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')  # add vertical line at x=0.0 (y axis drawed)

# Draw a horizontal span (rectangle) from ymin to ymax. 
# With the default values of xmin = 0 and xmax = 1
# axhspan(ymin, ymax, xmin=0, xmax=1, hold=None, **kwargs)
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k') # add horizontal line at y=0.5
plt.yticks([0.0, 0.5, 1.0]) # set the label 0, 0.5 and 1 on y axis
plt.ylim(-0.1, 1.1) # see the visible limit of the y axis
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()