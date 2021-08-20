#code from colab,

import os
import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

#设置
num_epochs = 90 #迭代次数
bath_size = 128 #批量大小
learning_rate = 0.01 #学习率大小
num_classes = 1000 #分类大小



class AlexNet(nn.Module):
  def __init__(self, num_outputs):
    super().__init__()
    self.net = nn.Sequential(
      #卷积层
      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=5), #section 3.3
      nn.MaxPool2d(kernel_size=3, stride=2),
      
      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(284, 284, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Flatten(),
      #全连接层
      nn.Linear(in_features=(256*6*6), out_features=4096),
      nn.ReLU(),
      nn.Dropout(p=0.5), #section4.2
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(4096, num_outputs)
    )
    self.init_weights()
    self.init_bias()


  def init_bias(self):
    for layer in self.net:
      if isinstance(layer, nn.Conv2d):
        nn.init.zeros_(layer.bias)
      if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.bias, 1)
    #2th, 4th, 5th convolutional layers
    nn.init.constant_(self.net[4].bias, 1)
    nn.init.constant_(self.net[10].bias, 1)
    nn.init.constant_(self.net[12].bias, 1)
  
  def init_weights(self):
    for layer in self.net:
      if type(layer)==nn.Linear or type(layer)==nn.Conv2d:
        nn.init.normal_(layer.weight, 0, 0.01)


#load dataset, 数据集很大
#TODO


class Accumulator:
  def __init__(self, n):
    self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

#计算准确率
def accuracy(y_hat, y):
  """计算正确预测的数量"""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#用于计算测试集的准确率
def evaluate_accuracy_gpu(net, data_iter, device=None):
  if isinstance(net, nn.Module):
    net.eval() # Set the model to evaluation mode
  metric = Accumulator(2)
  with torch.no_grad():
    for X, y in data_iter:
      if isinstance(X, list):
        X = [x.to(device) for x in X]
      else:
        X = X.to(device)
      y = y.to(device)
      metric.add(accuracy(net(X), y), y.numel())
  return metric[0]/metric[1]

#使用gpu
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train(net, train_iter, test_iter, num_epochs, lr, device):
  print('training on', device)
  optimizer = torch.optim.SGD(net.parameters(), lr=lr) #原文使用Momentum优化
  loss = nn.CrossEntropyLoss()
  for epoch in range(num_epochs):
    #sum of train loss, sum of train accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
      optimizer.zero_grad()
      X, y = X.to(device), y.to(device)
      y_hat = net(X)
      l = loss(y_hat, y)
      l.backward()
      optimizer.step()
      with torch.no_grad():
        metric.add(l*X.shape[0], accuracy(y_hat,y), X.shape[0])
      train_l = metric[0]/metric[2]
      train_acc = metric[1]/metric[2]
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'epoch {epoch},\t loss {train_l:.3f}, train accuracy {train_acc:.3f}, test accuracy {test_acc:.3f}')


if __name == __main__:

    #load data
    train_iter, test_iter = ....
    
    #create AlexNet
    alex_net = AlexNet(num_classes)
    #train
    train(alex_net, train_iter, test_iter, num_epochs, learning_rate, try_gpu())

    #


