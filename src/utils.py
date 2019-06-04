from data import DriverSequenceDataset
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

# get names of data files we want to load
def get_data_filenames(sequence_window_secs, crash_window):
  file_prefix = 'seq_len_' + str(sequence_window_secs) + '_window_' + str(crash_window[0]) + '_' + str(crash_window[1])
  data_filename = file_prefix + '_data.pt'
  labels_filename = file_prefix + '_labels.pt'
  sequence_info_filename = file_prefix + '_info.txt'
  return (data_filename, labels_filename, sequence_info_filename)

# get name of experiments folder
def get_experiments_folder_name(sequence_window_secs, crash_window):
  file_prefix = 'seq_len_' + str(sequence_window_secs) + '_window_' + str(crash_window[0]) + '_' + str(crash_window[1]) + '/'
  return file_prefix

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

def get_F_n_score(n, precision, recall):
  return (1 + n ** 2) *  (precision * recall) / (n ** 2 * precision + recall + 1e-5)

def check_accuracy(model, data, print_stats):
  tp, fp, tn, fn, correct, num_crashes, num_preds, crashes_predicted = 0, 0, 0, 0, 0, 0, 0, 0
  with torch.no_grad():
      for (val_X_batch, val_y_batch) in data:
          val_X_batch = val_X_batch.float()
          val_y_batch = val_y_batch.long()
          output = model(val_X_batch)
          output = torch.squeeze(output, 0)
          predictions = torch.argmax(output, 1)
          btp, bfp, btn, bfn, bcorrect, bnum_crashes, bnum_preds, bcrashes_predicted = metrics(predictions, val_y_batch)
          tp += btp
          fp += bfp
          tn += btn
          fn += bfn
          correct += bcorrect
          num_crashes += bnum_crashes
          num_preds += bnum_preds
          crashes_predicted += bcrashes_predicted

  precision = float(tp) / (tp + fp + 1e-5)
  recall = float(tp) / (tp + fn + 1e-5)
  # f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
  f1 = get_F_n_score(1, precision, recall)
  f2 = get_F_n_score(2, precision, recall)

  accuracy = float(correct) / num_preds
  if print_stats:
    print('overall accuracy: {}/{} ({}%)'.format(correct, num_preds, accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1: {}'.format(f1))
    print('f2: {}'.format(f2))
    print('total number of crashes: {}'.format(num_crashes))
    print('crashes predicted: {}'.format(crashes_predicted))
    print('')
  return f1, f2, precision, recall, accuracy

def load_numpy_data(seq_len, window_size, normalize=True):
  data_filename, labels_filename, sequence_info_filename = get_data_filenames(seq_len, window_size)
  root_dir = '../data/pytorch/'
  X = torch.load(root_dir + data_filename).numpy()
  y = torch.load(root_dir + labels_filename).numpy()

  N, T, D = X.shape
  if normalize:
    X_mean = np.mean(X, axis=(0, 1))
    X_std = np.std(X, axis=(0, 1))
    X = (X - X_mean) / X_std

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  return X_train, X_test, y_train, y_test

def load_pytorch_data(seq_len, window_size, batch_size=32):
  data_filename, labels_filename, sequence_info_filename = get_data_filenames(seq_len, window_size)

  data = DriverSequenceDataset(data_filename, labels_filename, '../data/pytorch/')
  train_size = int(0.8 * len(data))
  validation_size = len(data) - train_size
  train_data_split, validation_data_split = torch.utils.data.random_split(data, [train_size, validation_size])
  print(validation_data_split)
  print('Length of training data... {}'.format(train_size))
  print('Length of validation data... {}'.format(validation_size))

  train_data = DataLoader(train_data_split, batch_size, shuffle=True)
  validation_data = DataLoader(validation_data_split, batch_size, shuffle=True)
  return train_data, validation_data
