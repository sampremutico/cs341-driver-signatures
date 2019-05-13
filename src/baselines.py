from sklearn.ensemble import RandomForestClassifier
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

DATA_DIR = "../data/cs341-driver-data/nervtech/v1/drives-with-collisions/"

def load_data(train_split=0.8):
	X_ = None
	Y_ = None
	i=0
	for f in os.listdir(DATA_DIR):
		print('processing data for {}'.format(f))
		driver = DriverData(f, load=False)

		driver.segment_crashes(load=False)
		X, Y = driver.generate_sequences(as_np=True)
		num_rows, nx, ny = X.shape
		X = X.reshape((num_rows, nx * ny))

		if X_ is None:
			X_ = X
			Y_ = Y
		else:
			X_ = np.concatenate((X_, X))
			Y_ = np.concatenate((Y_, Y))
		break

	Y_ = Y_.astype(int)
	X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=1.0-train_split)

	# train_data = DataLoader(train_data_split, batch_size=8, shuffle=True)
	# validation_data = DataLoader(validation_data_split, batch_size=8, shuffle=True)
	return X_train, X_test, y_train, y_test

def score_model(model_name, model, X_train, X_test, y_train, y_test):
	model.fit(X_train, y_train)
	mean_accuracy = model.score(X_test, y_test)
	print('Mean accuracy for {name}: {score}'.format(name=model_name, score=mean_accuracy))

	y_pred = model.predict(X_test)
	print("Classification Report: {}".format(model_name))
	print(classification_report(y_test, y_pred, labels=[0, 1]))
	print("="*50)


if __name__ == '__main__':

	X_train, X_test, y_train, y_test = load_data(train_split=0.8)
	logistic_regression = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, \
							fit_intercept=True, intercept_scaling=1, class_weight=None, \
							random_state=None, solver='sag', max_iter=100, \
							verbose=0, warm_start=False, n_jobs=None)

	random_forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, \
						min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
						max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, \
						min_impurity_split=None, bootstrap=True, oob_score=False, \
						random_state=None, verbose=0, warm_start=False, class_weight=None)

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

	model_map = {'Logistic Regression': logistic_regression, 'Random Forest': random_forest, \
				'Support Vector Classification': svc, 'Decision Tree': decision_tree, \
				'Gaussian Process': gaussian_process, 'Multilayer Perceptron': multilayer_perceptron}

	for name, model in model_map.items():
		score_model(name, model, X_train, X_test, y_train, y_test)


