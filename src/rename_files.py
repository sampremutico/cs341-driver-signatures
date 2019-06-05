import os
from os.path import isfile, join, isdir

PATH = 'experiments/'

dirs = [f for f in os.listdir(PATH) if isdir(join(PATH, f))]

for dir in dirs:
  file_path_cnn = PATH + dir + '/cnn/'
  file_path_lstm = PATH + dir + '/lstm/'
  files_cnn = [f for f in os.listdir(file_path_cnn) if isfile(join(file_path_cnn, f))]
  files_lstm = [f for f in os.listdir(file_path_lstm) if isfile(join(file_path_lstm, f))]
  for f in files_cnn:
    with open(file_path_cnn + f, 'r') as l:
      lines = l.readlines()
      lines = lines[-1].split('\t')
      f1_val = lines[1]
      os.rename(file_path_cnn + f, file_path_cnn + f1_val)
  for f in files_lstm:
    with open(file_path_lstm + f, 'r') as l:
      lines = l.readlines()
      lines = lines[-1].split('\t')
      f1_val = lines[1]
      os.rename(file_path_lstm + f, file_path_lstm + f1_val)

print('done!')