from load_data import load_pytorch_data
from simple_lstm import SimpleLSTM
import torch
import numpy as np
import matplotlib.pyplot as plt

train_data, validation_data = load_pytorch_data(batch_size=32)

# TODO: Move this to some utils file
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

train_data, validation_data = load_data()

INPUT_SIZE = 76
hidden_size = 50

model = SimpleLSTM(INPUT_SIZE, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_weights = torch.tensor([0.1, 0.9])
criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
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


    if iter % 40 == 0:
      print('Iter {} loss: {}'.format(iter, loss.item()))
      tp, fp, tn, fn, correct, num_crashes, num_preds, crashes_predicted = 0, 0, 0, 0, 0, 0, 0, 0
      with torch.no_grad():
          for (val_X_batch, val_y_batch) in validation_data:
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

      precision = float(tp) / (tp + fp)
      recall = float(tp) / (tp + fn)
      accuracy = float(correct) / num_preds
      print('validation overall accuracy: {}/{} ({}%)'.format(correct, num_preds, accuracy))
      print('precision: {}'.format(precision))
      print('recall: {}'.format(recall))
      print('total number of crashes: {}'.format(num_crashes))
      print('crashes predicted: {}'.format(crashes_predicted))
      print('')

x = [i for i in range(len(losses))]
plt.plot(x, losses)
plt.show()

print('finished training!')





