from data import DriverSequenceDataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)


def metrics(y_pred, y_true):
  y_pred, y_true = y_pred.numpy(), y_true.numpy()
  assert(len(y_pred) == len(y_true))
  num_preds = len(y_pred)
  tp, fp, tn, fn = 0., 0., 0., 0.
  correct = 0
  num_crashes = 0
  crashes_predicted = 0
  for i in range(num_preds):
    if y_pred[i] == y_true[i] == 1:
      tp += 1
    elif y_pred[i] == 1 and y_pred[i] != y_true[i]:
      fp += 1
    elif y_pred[i] == y_true[i] == 0:
      tn += 1
    elif y_pred[i] == 0 and y_pred[i] != y_true[i]:
      fn += 1
    if y_pred[i] == y_true[i]:
      correct += 1
    if y_true[i] == 1:
      num_crashes += 1
    if y_pred[i] == 1:
      crashes_predicted += 1


  return tp, fp, tn, fn, correct, num_crashes, num_preds, crashes_predicted

def load_numpy_data():
  root_dir = '../data/pytorch/'
  file_X = 'data.pt'
  file_y = 'labels.pt'
  X = torch.load(root_dir + file_X).numpy()
  y = torch.load(root_dir + file_y).numpy()

  N, T, D = X.shape
  X_flattened = X.reshape((N * T, D))

  X_mean = np.mean(X_flattened, axis=0, keepdims=True)
  X_std = np.std(X_flattened, axis=0, keepdims=True)
  X = (X - X_mean) / X_std

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  return X_train, X_test, y_train, y_test

def load_pytorch_data(batch_size=32):
  data = DriverSequenceDataset('data.pt', 'labels.pt', '../data/pytorch/')
  train_size = int(0.8 * len(data))
  validation_size = len(data) - train_size
  train_data_split, validation_data_split = torch.utils.data.random_split(data, [train_size, validation_size])
  print('Length of training data... {}'.format(train_size))
  print('Length of validation data... {}'.format(validation_size))

  train_data = DataLoader(train_data_split, batch_size, shuffle=True)
  validation_data = DataLoader(validation_data_split, batch_size, shuffle=True)
  return train_data, validation_data
