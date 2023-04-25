import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.layer_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
		self.layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
		self.layer_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
		self.layer_7 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.layer_8 = nn.Dropout(p=0.25)
		self.layer_10 = nn.Linear(in_features=7744, out_features=128)
		self.layer_12 = nn.Dropout(p=0.5)
		self.layer_13 = nn.Linear(in_features=128, out_features=10)

	def forward(self, x):
		x = self.layer_1(x)
		x = F.relu(x)
		x = self.layer_3(x)
		x = F.relu(x)
		x = self.layer_5(x)
		x = F.relu(x)
		x = self.layer_7(x)
		x = self.layer_8(x)
		x = x.view(64, -1)
		x = self.layer_10(x)
		x = F.relu(x)
		x = self.layer_12(x)
		x = self.layer_13(x)
		x = F.log_softmax(x, dim=1)
		return x
