import torch

class SimpleLSTM(torch.nn.Module):
  def __init__(self, D_in, D_h, C=2):
    super(SimpleLSTM, self).__init__()
    self.lstm = torch.nn.LSTM(D_in, D_h, batch_first=True)
    
    for name, param in self.lstm.named_parameters():
    	if 'bias' in name:
    		torch.nn.init.constant(param, 0.0)
    	elif 'weight' in name:
    		torch.nn.init.xavier_normal(param)

    self.projection = torch.nn.Linear(D_h, C)

  def forward(self, x):
    output, (h_n, c_n) = self.lstm(x)
    scores = self.projection(h_n)
    return scores

