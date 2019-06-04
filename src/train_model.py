from models import SimpleLSTM, SimpleCNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_pytorch_data, metrics, check_accuracy
import copy

INPUT_SIZE = 76
SEQ_SIZE = 150
HZ = 30

#TODO: write to file, checkpoint
def train(seq_len, window_size, model_type, params, batch_size=16, num_epochs=20, print_every=10):
  metrics = []
  max_val_f2_score = 0.
  best_model = None

  train_data, validation_data = load_pytorch_data(seq_len, window_size)
  if model_type == 'LSTM':
    model = SimpleLSTM(INPUT_SIZE, params['lstm_hidden_size'])
  elif model_type == 'CNN':
    SEQ
    model = SimpleCNN(HZ * seq_len, params['cnn_hidden_size'])
  else:
    raise Exception('invalid model type')
  optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
  criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(params['loss_weights']))
  print('starting training!')
  for epoch in range(num_epochs):
    print('starting epoch {}...'.format(epoch))
    for iter, (X_batch, y_batch, idx) in enumerate(train_data):
      X_batch = X_batch.float()
      y_batch = y_batch.long()
      output = model(X_batch)
      output = torch.squeeze(output, 0)
      loss = criterion(output, y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if iter % print_every == 0:
        # print('Iter {} loss: {}'.format(iter, loss.item()))
        f1_val, f2_val, precision_val, recall_val, accuracy_val = check_accuracy(model, validation_data, False)
        f1_train, f2_train, precision_train, recall_train, accuracy_train = check_accuracy(model, train_data, False)
        train_loss = loss.item()
        metrics.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(train_loss, f1_val, f2_val, precision_train, recall_val, accuracy_val, f1_train, f2_train, precision_train, recall_train, accuracy_train))

        if f2_val > max_val_f2_score:
          max_val_f2_score = f2_val
          best_model = copy.deepcopy(model)

  print('finished training!')
  return best_model, max_val_f2_score, metrics








