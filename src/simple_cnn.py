import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, pool_size=2, num_classes=2, num_layers=1):
		super(SimpleCNN, self).__init__()
		self.input_size = input_size
		# A.K.A num output channels = num filters to apply
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_classes = num_classes
		# TODO: PADDING??
		self.conv1 = nn.Conv1d(input_size, hidden_size[0], kernel_size=3, padding=1)
		self.pool_1 = nn.MaxPool1d(kernel_size=2)
		self.conv2 = nn.Conv1d(hidden_size[0]/float(pool_size), hidden_size[1], kernel_size=3, padding=1)
		self.pool_2 = nn.MaxPool1d(kernel_size=2)

		self.relu = F.relu

		self.projection = nn.Linear(8000, self.num_classes)
		
	def forward(self, x):
		print("Input : {}".format(x.shape))
		x_conv_1_out = self.conv1(x).permute(0,2,1)
		print("Conv1 : {}".format(x_conv_1_out.shape))
		x_pool_1 = self.pool_1(x_conv_1_out).permute(0,2,1)
		print("Pool1 : {}".format(x_pool_1.shape))
		x_layer_1_out = self.relu(x_pool_1)

		x_conv_2_out = self.conv2(x_layer_1_out).permute(0,2,1)
		print("Conv2 : {}".format(x_conv_2_out.shape))
		x_pool_2 = self.pool_2(x_conv_2_out).permute(0,2,1)
		print("Pool2 : {}".format(x_pool_2.shape))
		x_layer_2_out = self.relu(x_pool_2)

		N, F, C = x_layer_2_out.shape
		x_final = x_layer_2_out.reshape(1,N*F*C)

		print("Reshaped : {}".format(x_final.shape))

		scores = self.projection(x_final)
		
		return scores