from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
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
		X, Y = driver.generate_sequences()
		num_rows, nx, ny = X.shape
		X = X.reshape((num_rows, nx * ny))

		if X_ is None:
			X_ = X
			Y_ = Y
		else:
			X_ = np.concatenate((X_, X))
			Y_ = np.concatenate((Y_, Y))

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
	logistic_regression = LogisticRegression()
	random_forest = RandomForestClassifier()
	svc = SVC()
	decision_tree = DecisionTreeClassifier()
	gaussian_process = GaussianProcessClassifier()

	model_map = {'Logistic Regression': logistic_regression, 'Random Forest': random_forest, \
				'Support Vector Classification': svc, 'Decision Tree': decision_tree} #, 'Gaussian Process': gaussian_process}

	for name, model in model_map.items():
		score_model(name, model, X_train, X_test, y_train, y_test)


