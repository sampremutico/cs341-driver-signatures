import torch
from torch.utils.data import Dataset
import numpy as np

class DriverSequenceDataset(Dataset):
  def __init__(self, file_X, file_y, root_dir, normalize=True):
    self.X = torch.load(root_dir + file_X)
    self.y = torch.load(root_dir + file_y)

    if normalize:
      N, T, D = self.X.size()
      X_flattened = self.X.view(N*T, D)
      mean = torch.mean(X_flattened, dim=0, keepdim=True)
      std = torch.std(X_flattened, dim=0, keepdim=True)
      X_normalized = (X_flattened - mean)/std
      self.X = X_normalized.view(N, T, D)

  def __len__(self):
    return self.X.size()[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.y[idx], idx)

