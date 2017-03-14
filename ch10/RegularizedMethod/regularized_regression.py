# as seen in ch3, regularization is one approach to tackle the
# problem of overfitting by adding additional info and so shrink
# the param values of the model

# most know :
	# Ridge Regression : J(w) = Σ(y(i) - ŷ(i))² + λ||w||²2 (L2 norm)
	# Least Absolute Shrinkage and Selection Operator (LASSO) (L1 norm)
	# Elastic Net (mix of 2 above) : J(w) = Σ(y(i) - ŷ(i))² + λ1||w||²2 + λ2|w|

	# it works similarly to the LinearRegression() but we need to
	# specify the param lambda...

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)

from sklearn.linear_model import ElasticNet
lasso = ElasticNet(alpha=1.0, l1_ratio=0.5)