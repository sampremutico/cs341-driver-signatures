from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from data import DriverSequenceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from DriverData import DriverData
import matplotlib.pyplot as plt
import pandas as pd
import torch, os
from utils import load_numpy_data
import json


# import xgboost as xgb

def load_column_names():
	with open('column_names.json', 'r') as f:
	    col_map = json.load(f)
	    assert(type(col_map)==dict)
	    col_map = { int(k): v for k,v in col_map.items() }
	    f.close()
	    return col_map

def score_model(model_name, model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	mean_accuracy = model.score(X_test, y_test)
	print('Mean accuracy for {name}: {score}'.format(name=model_name, score=mean_accuracy))

	y_pred = model.predict(X_test)
	print("Classification Report: {}".format(model_name))
	print(classification_report(y_test, y_pred, labels=[0, 1]))
	print("="*50)

	if model_name == 'Random Forest': random_forest_importance(model, X_train, num_features_to_plot=20)

def random_forest_importance(model, X_train, num_features_to_plot=None):
	if num_features_to_plot is None: num_features_to_plot = X_train.shape[1]

	importances = model.feature_importances_
	std = np.std([mod.feature_importances_ for mod in model.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	col_map = load_column_names()
	print("Feature ranking:")

	features = []
	for f in range(num_features_to_plot):
		feature = col_map[indices[f]]
		print("%d. feature %s (%f)" % (f + 1, feature, importances[indices[f]]))
		features.append(feature.lower())

	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(num_features_to_plot), importances[indices][:num_features_to_plot], color="r", yerr=std[indices][:num_features_to_plot], align="center")
	plt.xticks(range(num_features_to_plot), features, fontsize=8)
	plt.xlim([-1, num_features_to_plot])
	plt.xticks(rotation=80)
	plt.show()

if __name__ == '__main__':

	X_train, X_test, y_train, y_test = load_numpy_data()
	print('X_train shape: {}'.format(X_train.shape))
	print('y_train shape: {}'.format(y_train.shape))
	print('X_test shape: {}'.format(X_test.shape))
	print('y_test shape: {}'.format(y_test.shape))


	print("Averaging across all {} timesteps per sequence".format(X_train.shape[1]))
	X_train = np.mean(X_train, axis=1)
	X_test = np.mean(X_test, axis=1)

	# X_train = X_train.reshape(X_train.shape[0], -1)
	# X_test = X_test.reshape(X_test.shape[0], -1)

	logistic_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, \
							fit_intercept=True, intercept_scaling=1, class_weight=None, \
							random_state=None, solver='sag', max_iter=200, \
							verbose=0, warm_start=False, n_jobs=None)

	random_forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, \
						min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
						max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, \
						min_impurity_split=None, bootstrap=True, oob_score=False, \
						random_state=None, verbose=0, warm_start=False, class_weight=None)

	gradient_boosting = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, \
							subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,\
							min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, \
							init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, \
							presort='auto')

	svc = SVC()

	decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, \
						min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
						max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, \
						min_impurity_split=None, class_weight=None, presort=False)

	gaussian_process = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, \
						max_iter_predict=100, warm_start=False, copy_X_train=True, \
						random_state=None, multi_class='one_vs_one', n_jobs=None)

	multilayer_perceptron = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', \
								solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', \
								learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, \
								random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, \
								nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, \
								beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model_map = {'Logistic Regression': logistic_regression, 'Gradient Boosting': gradient_boosting, \
				'Support Vector Classification': svc, 'Decision Tree': decision_tree, \
				'Gaussian Process': gaussian_process, 'Multilayer Perceptron': multilayer_perceptron}

	score_model('Random Forest', random_forest, X_train, X_test, y_train, y_test)
	for name, model in model_map.items():
		score_model(name, model, X_train, X_test, y_train, y_test)

