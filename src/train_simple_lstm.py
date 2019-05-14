from simple_lstm import SimpleLSTM
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_pytorch_data, metrics, check_validation_accuracy


INPUT_SIZE = 76

def train(batch_size=16, lstm_hidden_size=50, lr=1e-4, loss_weights=[0.1, 0.9], print_every=20, checkpointing=False):
  train_data, validation_data = load_pytorch_data(batch_size)
  model = SimpleLSTM(INPUT_SIZE, lstm_hidden_size)
  optimizer = torch.optim.Adam(model.parameters(), lr)
  criterion = torch.nn.CrossEntropyLoss(torch.tensor(loss_weights))
  losses = []
  print('starting training!')
  for epoch in range(20):
    print('starting epoch {}...'.format(epoch))
    for iter, (X_batch, y_batch) in enumerate(train_data):
      X_batch = X_batch.float()
      y_batch = y_batch.long()
      output = model(X_batch)
      output = torch.squeeze(output, 0)
      loss = criterion(output, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if iter % print_every == 0:
        print('Iter {} loss: {}'.format(iter, loss.item()))
        losses.append(loss.item())
        f1 = check_validation_accuracy(model, validation_data)

  x = [i for i in range(len(losses))]
  plt.plot(x, losses)
  plt.show()

  print('finished training!')



# TODO:
# Plot loss curves, validation accuracy/f1 scorees
# hyperparameter tuning
# hidden size, learning rate, loss weights
# optimizer?
# checkpoint models

train(batch_size=16, lr=1e-3)
train(batch_size=16, lr=1e-4)
train(batch_size=16, lr=1e-5)