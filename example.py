#imageNet数据集过大，考虑到训练时间问题以及硬件设备不足
#使用简化的AlexNet训练Fashion-MNIST数据集

#pip install d2l
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

batch_size = 128
num_epochs = 10
learning_rate = 0.01



def load_data_fashion_mnist(batch_size, resize=None):
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)

  mnist_train = torchvision.datasets.FashionMNIST(root='../data',
                            train=True,
                            transform=trans,
                            download=True)
  mnist_test = torchvision.datasets.FashionMNIST(root='../data',
                              train=False,
                              transform=trans,
                              download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True),
       data.DataLoader(mnist_test, batch_size, shuffle=False))

#fashion-mnist只有一个通道 
#构建一个适合fashion-mnist的alexNet神经网络
net = nn.Sequential(
    #卷积层
    nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
    nn.ReLU(),
    #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=5), #section 3.3
    nn.MaxPool2d(kernel_size=3, stride=2),
      
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(), #6400
    #全连接层
    nn.Linear(in_features= 6400, out_features=4096),
    nn.ReLU(),
    nn.Dropout(p=0.5), #section4.2
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10) # 10个分类输出
)


def try_gpu(i=0):
  """Return gpu(i) if exists, otherwise return cpu()."""
  if torch.cuda.device_count() >= i + 1:
    return torch.device(f'cuda:{i}')
  return torch.device('cpu')

train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224) #batch_size=128
print(train_iter)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, learning_rate, try_gpu())