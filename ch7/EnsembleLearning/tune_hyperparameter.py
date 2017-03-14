# we can access to the parameters of our newly created classifier
# (the one we created using MajorityVoting) defined in decision_region
# using the fct we defined in majority_vote : .get_params()

# let's tune the inverse reg parameter C of the logistic regression
# classifier and the decision tree depth via a gridsearch.

# first we import the dependecies and all the variables

from decision_region import *

# find the parameter using :
# mv_clf.get_params()


# Note : GridSearch will perfom a brute force to find the params
# that lead to the best result.
from sklearn.grid_search import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1, 2],
		  'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

# initialize the parameters of the grid search
grid = GridSearchCV(estimator=mv_clf,
				   param_grid=params,
				   cv=10, # 10-fold stratified cross-validation
				   scoring='roc_auc') # can choose accuracy or other

# fit the data to our classifier to train it
grid.fit(X_train, y_train)

# print the different hyperparameter value combinations and
# the average ROC AUC scores computed via 10-fols cross-validation

# params : a dict of parameter settings
# mean_score : the mean score over the cross-validation folds
# scores : the list of scores for each fold

for params, mean_score, scores in grid.grid_scores_:
	print("%0.3f+/-%0.2f %r"
			% (mean_score, scores.std() / 2, params))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)