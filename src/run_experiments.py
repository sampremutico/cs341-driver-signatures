from train_model import train
import numpy as np
from utils  import get_experiments_folder_name
import argparse
import os

EXPERIMENTS_PATH = 'experiments/'

def write_metrics_to_file(max_f1_score, metrics, params, path):
  filename = str(round(max_f1_score, 8))
  with open(path + filename, 'w') as f:
    for key in sorted(params.keys()):
      f.write(key + '\t' + str(params[key]) + '\n')
    for metric in metrics:
      f.write(metric + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
  parser.add_argument('--window_s', type=int, required=True, help='Window start')
  parser.add_argument('--window_e', type=int, required=True, help='Window start')
  args = parser.parse_args()

  seq_len = args.seq_len
  window_size = (args.window_s, args.window_e)

  # num_lstm_experiments, num_cnn_experiments = 100, 100

  num_lstm_experiments, num_cnn_experiments = 1, 1

  loss_weight_class1_vals = [0.75]
  lr = 	0.000275271739294
  lstm_hidden_sizes = [60]

  # lstm_hidden_sizes = [20, 40, 60, 80]
  # cnn_hidden_sizes = [20, 40, 60, 80]
  # loss_weight_class1_vals = np.linspace(.6, .95, 15)





  experiments_folder_name = get_experiments_folder_name(seq_len, window_size)
  CNN_PATH = EXPERIMENTS_PATH + experiments_folder_name + 'cnn/'
  LSTM_PATH = EXPERIMENTS_PATH + experiments_folder_name + 'lstm/'

  if not os.path.exists(os.path.dirname(CNN_PATH)): os.makedirs(CNN_PATH)
  if not os.path.exists(os.path.dirname(LSTM_PATH)): os.makedirs(LSTM_PATH)

  # for i in range(num_cnn_experiments):
  #   print('Running CNN experiment {}/{}'.format(i + 1, num_cnn_experiments))
  #   cnn_hidden_size_layer1 = np.random.choice(cnn_hidden_sizes)
  #   cnn_hidden_size = (cnn_hidden_size_layer1, int(cnn_hidden_size_layer1 / 2))
  #   lr = 10 ** np.random.uniform(-6, -2)
  #   loss_weight_class1 = np.random.choice(loss_weight_class1_vals)
  #   loss_weights = [1.-loss_weight_class1, loss_weight_class1]
  #   params = {
  #     'cnn_hidden_size': cnn_hidden_size,
  #     'lr': lr,
  #     'loss_weights': loss_weights
  #   }
  #   print(params)
  #   best_model, max_f1_score, metrics = train(seq_len, window_size, 'CNN', params)
  #   write_metrics_to_file(max_f1_score, metrics, params, CNN_PATH)
  #   print('')


  for i in range(num_lstm_experiments):
    print('Running LSTM experiment {}/{}'.format(i + 1, num_lstm_experiments))
    lstm_hidden_size = np.random.choice(lstm_hidden_sizes)
    # lr = 10 ** np.random.uniform(-6, -2)
    loss_weight_class1 = np.random.choice(loss_weight_class1_vals)
    loss_weights = [1.-loss_weight_class1, loss_weight_class1]
    params = {
      'lstm_hidden_size': lstm_hidden_size,
      'lr': lr,
      'loss_weights': loss_weights
    }
    print(params)
    best_model, max_f1_score, metrics = train(seq_len, window_size, 'LSTM', params)
    write_metrics_to_file(max_f1_score, metrics, params, LSTM_PATH)
    print('')





