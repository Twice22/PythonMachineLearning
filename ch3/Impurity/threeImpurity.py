import matplotlib.pyplot as plt
import numpy as np 

def gini(p):
	return (p)*(1-(p)) + (1-p)*(1-(1-p))

def entropy(p):
	return - p*np.log2(p) - (1-p)*np.log2((1-p))

def error(p):
	return 1 - np.max([p, 1-p])

x = np.arange(0.0, 1.0, 0.01) # array from 0 to 1 each 0.01 step
# ent = [None, entropy(0.01), entropy(0.02), ...]
ent = [entropy(p) if p != 0 else None for p in x]
# sc_ent = [None, 0.1*ent[1], 0.2*ent[2], ...]
sc_ent = [e*0.5 if e else None for e in ent]
# err = [error(0), error(1), ...]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip ([ent, sc_ent, gini(x), err],
							['Entropy', 'Entropy (scaled)',
							 'Gini Impurity', 'Misclassification Error'],
							 ['-', '-', '--', '-.'],
							 ['black', 'lightgray', 'red', 'green', 'cyan']):
	line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
		  ncol=3, fancybox=True, shadow=False)


ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1]) # the view will go from 0 to 1.1 on y axis
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.tight_layout()

plt.show()