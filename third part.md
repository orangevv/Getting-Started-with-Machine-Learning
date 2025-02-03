Here is your content converted to markdown format:

```markdown
# Pytorch教程：DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ

为了大家观看方便，我在这里直接做了一个四个部分内容的跳转，大家可以自行选择观看。

- [第一部分](#第一部分)
- [第二部分](#第二部分)
- [第三部分](#第三部分)
- [第四部分](#第四部分)

Pytorch官网有非常优秀的教程，其中有几篇小短文属于名为 **DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ** 这个小专栏的内容，考虑到大家阅读英文文献有点困难，笔者打算花些时间做一下翻译，同时结合自己的理解做一些内容调整，原文链接贴在这里[点此跳转](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。承接之前的内容，点此跳转到[第二部分](#第二部分)。好的我们开始[第三部分](#第三部分)。

## 第三部分：神经网络

这一部分的内容是神经网络，这一次就开始对整个网络进行介绍，为下一部分的实战做一个铺垫。

神经网络可以使用`torch.nn`的包来构建。

如今你已经了解过一些autograd的内容，`nn`依赖于autograd来定义和区分模型。一个`nn.Module`包含很多层，包括前向从输出开始到返回输出。

举个栗子，看一下下面这一张通过网络分类图片的图：

它是一个简单的前馈神经网络，它通过获取输入，喂给一层又一层的网络，然后最后输出结果。

### 经典的网络训练过程

一个经典的网络训练过程如下图：

1. 定义一个神经网络然后有一些可以学习的参数或者权重
2. 遍历输入数据集
3. 通过神经网络输入数据
4. 计算loss值（预测值和真实值的误差）
5. 将梯度回传到网络参数中
6. 更新网络权重，通常使用一个简单的更新规则：`weight = weight - learning_rate * gradient`

### 定义一个网络

让我们定义这个网络：

```python
# 引入对于的库
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1个图片输入通道, 6个输出通道, 5x5 面积的卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池化层通过了一个2*2的窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # 除batch（批量）使用的维度外的所有尺寸都要打平，即把高维降成一维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
```

结果输出：

```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

你必须定义前向传播，然后反向传播函数（用来计算梯度）被使用`autograd`自动地定义。你可以使用任意一个tensor操作在前向传播函数中。

### 模型的学习率参数

一个模型的学习率参数通过`net.parameters()`来学习。

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1的权重
```

输出结果为：

```
10
torch.Size([6, 1, 5, 5])
```

让我们试着随机化32 × 32的输入，这个网络（LeNet）的输入尺寸是32 × 32。为了在MNIST上使用数据集，需要改变数据集图片的尺寸为32 × 32。

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

输出结果为：

```
tensor([[-0.0794,  0.0241,  0.0712, -0.0940,  0.0481, -0.0220,  0.0628,  0.0115,
         -0.0880, -0.0059]], grad_fn=<AddmmBackward>)
```

### 用零初始化所有参数的梯度，然后用随机数初始化反向传播的参数

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

`torch.nn`仅仅支持mini-batch（小批量的，就是把大的数据集分成一批一批的），整个`torch.nn`包紧急支持输出一个mini-batch的样本，而不支持以后单独的样本。

举个例子，`nn.Conv2d`将采用一个4个维度的tensor，`nSample × nChannels × Height × Width`

如果你有一个单独的样本，使用`input.unsqueeze(0)`去伪装成一批数据。

### 回顾

在继续学习其他知识之前，让我们重新回顾已经学习的知识。

1. **torch.Tensor**: 一个支持自动求梯度操作的多维度的数组就像`backward()`。也会保存相关梯度的参数。
2. **nn.Module**: 神经网络模型，方便参数封装，可以移动到GPU中，读取和保存。
3. **nn.Parameter**: 一种tensor，能够自动化注册参数然后分配给对应的一个模块上。
4. **autograd.Function**: 实现前向或者反向创博使用自动化执行的方式，创建一个tensor然后对它的历史过程进行编码。

### 损失函数（Loss Function）

一个损失函数获取（输出结果，目标结果）输入对，然后计算一个数值，预估输出结果和目标结果相差多少。

有许多不同的损失函数都在`nn`这个包中。一个简单的loss是：`nn.MSELoss`，这个可以计算两个数据直接的均方差。

```python
output = net(input)
target = torch.randn(10)  # 一个假设的目标
target = target.view(1, -1)  # 使用同样的形状作为输出
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

输出结果为：

```
tensor(0.8815, grad_fn=<MseLossBackward>)
```

如今，给你一个loss值，使用它的`.grad_fn`就可以看到图表化的计算结果：

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

所以，当我们调用`loss.backward()`的时候，整个图会区分神经网络的各个参数，所有的tensor带有`requires_grad=True`的结果都会有他们自己`.grad` tensor结果。

### 反向传播

为了反向传播误差值我们需要去使用`loss.backward()`。你需要去清除现在已经存在的梯度值的梯度，否则梯度会不断累积。

现在我们应该调用`loss.backward()`，然后找到`conv1`的偏置梯度然后反向传播。

```python
net.zero_grad()  # 用零初始化所有参数的地图

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

输出结果为：

```
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0029, -0.0122,  0.0044,  0.0115,  0.0076,  0.0122])
```

### 更新权重

最简单的更新规则就是使用随机梯度下降法（Stochastic Gradient Descent,简称SGD）:

```python
weight = weight - learning_rate * gradient
```

我们能够实现这种简单的Python代码：

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

但是如果你需要使用一个神经网络，你想去适应各种各样的更新规则比如SGD，Nestrov-SGD，Adam，RMSProp等等，可以使用这个`torch.optim`这个包。

```python
import torch.optim as optim

# 创建你的优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

这第三部分也就结束了，接下来就等着最后一部分了，我尽快。
```
