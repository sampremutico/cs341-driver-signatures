import torch
from torch.utils.data import Dataset

class DriverSequenceDataset(Dataset):
  def __init__(self, file_X, file_y, root_dir):
    self.X = torch.load(root_dir + file_X)
    self.y = torch.load(root_dir + file_y)

  def __len__(self):
    return self.X.size()[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.y[idx])

