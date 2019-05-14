import torch
from torch.utils.data import Dataset
import numpy as np

class DriverSequenceDataset(Dataset):
  def __init__(self, file_X, file_y, root_dir):
    self.X = torch.load(root_dir + file_X)
    self.y = torch.load(root_dir + file_y)


  def __len__(self):
    return self.X.size()[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.y[idx])

  def get_X(self, np):
    if np: return self.X.numpy()
    return self.X

  def get_Y(self, np):
    if np: return self.y.numpy()
    return self.y


