# 以下代码实现了一个简单的3层 卷积神经网络
# 代码来源：https://github.com/pytorch/examples/tree/master/mnist

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d() 参数依次代表： in_channnel, out_channel, kernel_size, stride
        # nn.Conv2d() 表示一个卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # doopout 比较有效的缓解过拟合的发生
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # nn.Lineer() 全连接层 ，一般作为输出层，得到分类概率。fc2: 10 表示类别数
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # relu= max(0, x), 是一个非线性激活函数
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # max_pool2d 最大池化层
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # torch.flatten(x, start_dim, end_dim) 展平tensor x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # 这一步计算分类概率
        return output

