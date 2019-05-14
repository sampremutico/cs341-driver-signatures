from data import DriverSequenceDataset
from torch.utils.data import DataLoader
from simple_lstm import SimpleLSTM
import torch
import numpy as np

data = DriverSequenceDataset('data.pt', 'labels.pt', '../data/pytorch/')

# # should move this elsewhere
train_size = int(0.8 * len(data))
validation_size = len(data) - train_size
train_data_split, validation_data_split = torch.utils.data.random_split(data, [train_size, validation_size])
print('Length of training data... {}'.format(train_size))
print('Length of validation data... {}'.format(validation_size))

train_data = DataLoader(train_data_split, batch_size=8, shuffle=True)
validation_data = DataLoader(validation_data_split, batch_size=8, shuffle=True)

INPUT_SIZE = 103
hidden_size = 50

model = SimpleLSTM(INPUT_SIZE, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_weights = torch.tensor([0.05, 0.95])
criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
print('starting training!')

for epoch in range(10):
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

    if iter % 40 == 0:
      print('Iter {} loss: {}'.format(iter, loss.item()))
      correct = 0
      total = 0
      crashes_predicted = 0

      with torch.no_grad():
          for (val_X_batch, val_y_batch) in validation_data:
              val_X_batch = val_X_batch.float()
              val_y_batch = val_y_batch.long()
              output = model(val_X_batch)
              output = torch.squeeze(output, 0)
              predictions = torch.argmax(output, 1)
              total += predictions.size(0)
              correct += (predictions == val_y_batch).sum().item()
              crashes_predicted += (predictions == 1).sum().item()
      print('validation overall accuracy: {}/{} ({}%)'.format(correct, total, float(correct) / total))
      print('crashes predicted: {}'.format(crashes_predicted))
      print('')


print('finished training!')


