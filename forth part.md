Here is your content converted to markdown format:

```markdown
# Pytorch教程：DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ

为了大家观看方便，我在这里直接做了一个四个部分内容的跳转，大家可以自行选择观看。

- [第一部分](#第一部分)
- [第二部分](#第二部分)
- [第三部分](#第三部分)
- [第四部分](#第四部分)

Pytorch官网有非常优秀的教程，其中有几篇小短文属于名为 **DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ** 这个小专栏的内容，考虑到大家阅读英文文献有点困难，笔者打算花些时间做一下翻译，同时结合自己的理解做一些内容调整，原文链接贴在这里[点此跳转](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。承接之前的内容，点此跳转到[第三部分](#第三部分)。好的我们开始[第四部分](#第四部分)，这也是最后一个部分。

## 第四部分：实战

这一部分的内容让我们结合之前的知识进行简单的实战，学完这一部分就可以自己使用pytorch进行简单的模型搭建了，我们就直接开始了。

现在你已经知道了如何定义神经网络了，同时计算loss值并且更新神经网络的权重值。现在想请你思考一下。

### 数据是什么？

通常情况下，当你不得不去处理图片、文本、音频和视频数据时，你能使用标准的python包将数据导入到numpy数组中。然后你能够将数组转化为pytorch的tensor类型。

1. 对于图片，有Pillow，OpenCV的包可以使用。
2. 对于音频，有scipy和librosa的包可以使用。
3. 对于文本，无论是基于原始Python或Cython的加载，使用NLTK和SpaCy都是可以使用的。

特别的对于视觉的处理，我们已经创建一个包叫做`torchvision`，可以用于对公共数据集的数据加载程序，例如ImageNet, CIFAR10, MNIST等等。对于图片的数据转化工具包括`torchvision.datasets`和`torch.utils.data.DataLoader`。

这些工具为我们提供了很大的便利，避免了写过多的重复代码，便于相关人员的使用。

对于这篇教程，我们将会使用CIFAR10的数据集。它包含各个种类，比如‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’等。CIFAR10中的图片是3×32×32的大小，即3个（RGB）通道的32×32尺寸的图片。

### 训练一个图片分类器

我们将按照步骤进行如下的操作：

1. 使用`torchvision`导入并且规范化CIFAR10的训练数据和测试数据。
2. 定义一个卷积神经网络。
3. 定义一个损失（loss）函数。
4. 在训练数据集上训练一个神经网络。
5. 在测试集上测试网络的效果。

#### 1. 导入并且规范化CIFAR10的数据集

使用`torchvision`，用它来导入CIFAR10是非常简单的。

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 数据规范化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

如果在Windows上运行，你得到一个`BrokenPipeError`，那么可以将 `torch.utils.data.DataLoader()`的`num_workers`设置为0。

上图输出结果为：

```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
```

由于大家的计算机上还没有相关的数据集，所以下面会进行下载。

下面让我们显示一下进行训练的图片。

```python
import matplotlib.pyplot as plt
import numpy as np

# 显示一张图片的函数
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获得一些随机图片用作训练
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

输出结果为：

```
cat plane  bird  ship
```

#### 2. 定义一个卷积神经网络

从神经网络集中选取一个神经网络，调整网络的输入为一个3通道（指的是RGB）的图片，代替原来默认的单通道（灰度）的图片。

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 将高维度打平为一维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

#### 3. 定义一个损失函数和优化器

让我们使用一个分类器`Cross-Entropy`（交叉熵）的损失函数和带有动量的SGD（随机梯度下降法）。

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

#### 4. 训练网络

下面的事情开始变得有趣了，我们就是简单地让数据进行迭代循环，给网络喂输入数据然后让网络自行优化。

```python
for epoch in range(2):  # 整个数据集的循环次数
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # 零是元素梯度
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计数据
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个mini-batches打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

输出结果为：

```
[1,  2000] loss: 2.128
[1,  4000] loss: 1.793
[1,  6000] loss: 1.649
[1,  8000] loss: 1.555
[1, 10000] loss: 1.504
[1, 12000] loss: 1.444
[2,  2000] loss: 1.379
[2,  4000] loss: 1.344
[2,  6000] loss: 1.336
[2,  8000] loss: 1.327
[2, 10000] loss: 1.294
[2, 12000] loss: 1.280
Finished Training
```

#### 5. 在测试集上测试网络

我们已经在训练集上训练了2轮网络了（训练网络的那个位置大循环是2次），但是我们需要检验是否网络已经取得了不错的学习效果。

我们通过检测标签的分类情况和神经网络的输出结果，检测与真实值的误差。如果预测正确，我们会在正确预测集中增加一个样本。

首先显示来自测试集中的一组图片。

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# 打印图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

输出结果为：

```
GroundTruth:    cat  ship  ship plane
```

接下来，我们导入之前保存的模型，并测试网络。

```python
net = Net()
net.load_state_dict(torch.load(PATH))
```

接下来，让我们康康神经网络的输出。

```python
outputs = net(images)
```

输出是10个类别的概率，概率最高的那个类别的就认定为输出的结果就是那一个类别，所以我们会获取概率最高的那一个类别的索引。

```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```

输出结果为：

```
Predicted:   frog  ship  ship  ship
```

### 在GPU上训练

就像将tensor转换到GPU上一样，这里我们将网络转换到GPU上。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 假设我们有一个装有CUDA的机器应当打印对应机器设备代号:
print(device)
```

输出结果为：

```
cuda:0
```

这些方法会递归遍历所有模块，并将它们的参数和缓冲区转换为CUDA的tensor。

```python
net.to(device)
```

记住，你必须在每一步都将输入和目标发送给GPU参与运算。

```python
inputs, labels = data[0].to(device), data[1].to(device)
```

通过这次的学习，你可以建立一个小的网络进行图片的分类，下面你可以进一步理解PyTorch的库，然后训练更多的神经网络。
```
