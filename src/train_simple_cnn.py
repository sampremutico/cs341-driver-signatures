from load_data import load_data
from simple_cnn import SimpleCNN
import torch
import numpy as np

import matplotlib.pyplot as plt

train_data, validation_data = load_data()

INPUT_SIZE = 150#103
hidden_size = [80,40]

model = SimpleCNN(INPUT_SIZE, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_weights = torch.tensor([0.1, 0.9])
criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
print('starting training!')

losses = list()

for epoch in range(1):
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

    losses.append(loss.item)

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

plt.plot(range(len(losses)), losses)
plt.show()

print('finished training!')


