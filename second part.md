Here's your content converted to markdown format:

```markdown
# Pytorch教程：DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ

为了大家观看方便，我在这里直接做了一个四个部分内容的跳转，大家可以自行选择观看。

- [第一部分](#第一部分)
- [第二部分](#第二部分)
- [第三部分](#第三部分)
- [第四部分](#第四部分)

Pytorch官网有非常优秀的教程，其中有几篇小短文属于名为 **DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ** 这个小专栏的内容，考虑到大家阅读英文文献有点困难，笔者打算花两天时间做一下翻译，同时结合自己的理解做一些内容调整，原文链接贴在这里[点此跳转](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。承接之前的内容，点此跳转到[第一部分](#第一部分)。好的我们直接开始[第二部分](#第二部分)。

## 第二部分：一个平缓的对于pytorch自动求梯度的介绍

这里引入了一个函数`torch.autograd`，这是一个Pytorch用于增强神经网络训练的自动化区分引擎。在这个部分，你会在概念上完全理解autograd如何帮助一个神经网络进行训练。

### 背景

神经网络是一个嵌套的函数集合(就理解为大的函数，里面嵌套着一堆函数)，可以用来执行某些被输入到神经网络的数据。这些函数通过一系列参数定义，参数主要包括weights和biases即权重和偏置。这些在pytorch中都被存储在tensors上。

训练一个神经网络需要以下两个步骤：

1. **前向传播**：在前向传播中，卷积神经网络能够对正确的输出做出最精确的预测。神经网络通过它的函数运行输入数据给出它的猜测。

2. **反向传播**：在反向传播中，神经网络成比例的调整它的猜测与正确结果之间的误差。收集函数对函数参数中导数的误差，同时使用梯度下降优化参数。

### 使用Pytorch

让我们来看一下一个单步训练的过程，举个栗子，我们从`torchvision`中导入一个预先训练好的`resnet18`的模型。我们创建了一个随机数构成的tensor去代替一张图片的三个通道（当然就是RGB三个通道了），宽度取64，用随机值初始化对应的标签。

```python
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

下一步，我们将输入数据通过模型的每一层做出预测，这是前向传播。

```python
prediction = model(data)  # 前向传播
```

我们使用模型的预测值和对应的标签值去计算误差（也就是loss值，即误差值）。接下来是通过网络反向传播这个误差。在反向传播中自动求梯度然后在参数梯度属性中为每一个模型参数计算梯度，并且存储下来。

```python
loss = (prediction - labels).sum()
loss.backward()  # backward pass
```

下一步，我们引入一个优化器，在这个例子中SGD（随机梯度下降法）使用0.01的学习率和0.9的动量。我们在优化器中注册所有的模型参数。

最后我们使用`.step()`去初始化梯度下降。这个优化器适应每一个梯度存储在`.grad`中的情况。

```python
optim.step()  #梯度下降
```

在这一部分中，你需要去训练你的神经网络的每一个部分，下面一些部分的关于自动求梯度的一些细节，你可以直接跳过。

### 拆分autograd函数（自动求梯度）

让我们看一下autograd如何收集梯度。我们创建了两个tensor `a` 和 `b`，标记为 `requires_grad=True`，即可以求梯度。这个信号表示autograd的每一步操作都应该被跟踪。

```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
```

我们从`a`和`b`中创建另一个tensor `Q`。

```python
Q = 3*a**3 - b**2
```

让我们假设`a`和`b`是神经网络的参数，`Q`是loss值即真实值和神经网络预测值之间的误差值。在神经网络的训练中，我们想要知道loss值参数的梯度信息。

当我们在`Q`上使用`.backward()`函数的时候，autograd会计算这些梯度然后存储在各自的tensor的`.grad`属性中。

因为它是一个向量，`.gradient`是一个和`Q`一样形状的tensor，并且代表着`Q`本身的梯度，因此我们需要去清晰地传递一个在`Q.backward()`上的梯度参数。

同样的，我们也可以将`Q`聚合成一个标量并隐式地向后调用，就像`Q.sum().backward()`。

```python
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
```

梯度信息现在被放置在`a.grad`和`b.grad`中。

### 输出一下梯度看看是不是正确的

```python
print(9*a**2 == a.grad)
print(-2*b == b.grad)
```

输出结果：

```
tensor([True, True])
tensor([True, True])
```

### 可选读的部分 - 使用autograd计算向量微积分

在数学上，如果你有一个数值化的向量函数，然后梯度对求梯度变为Jacobin矩阵`J`：

通常情况下`torch.autograd`是一个计算机雅可比矩阵参数的一个引擎。给出任何一个向量，计算。

如果是标量函数的一个梯度：

然后通过导数的链式求解规则，雅可比矩阵的元素将会是对的导数。

雅可比矩阵的参数是就是我们显示在上图中的样子，`external_grad`代表着。

### 计算图

从概念上说，autograd保持数据(tensor)的记录，以及其他可执行的操作(和新tensor的结果一起)在一个由函数组成的有向无环图（简称DAG）中。在这个DAG中，叶子是输入的tensor，根结点是输出的tensor。通过从图的根节点跟踪到叶结点，你能使用链式法则自动地计算出梯度。

在一个前向传播中，autograd同时做了两件事情：

1. 运行请求的操作去计算一个tensor的结果
2. 在DAG中获取操作函数的梯度

当DAG的根节点调用`.backward()`的时候反向传播开始：

1. 计算每一个`.grad_fn`的梯度
2. 聚集他们在各自的tensor的`.grad`的属性
3. 使用链式法则，传播路径到叶子节点

下图是一个可视化的DAG图的样子。在这个图中，各个方向的箭头代表前向传播的方向。节点代表每一个操作的反向传播函数。叶子节点（蓝色的）代表叶子tensor `a` 和 `b`。

需要注意的一件重要的事情是，图是从头开始重新创建的。每次调用`.backward()`后，autograd开始填充一个新图。这正是允许你在模型中使用控制流语句的原因，你可以在每次迭代时更改形状、大小和操作。

### DAG图之外的细节

`torch.autograd`跟踪在所有的tensor上的操作，需要有一个`require_grad`的表示设置为`True`。对于不需要梯度的tensor，设置为`require_grad=False`，可以将其排除在DAG图之外。

尽管仅有一个输入tensor有`require_grad = True`，那么这个操作输出的tensor将需要梯度。

```python
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")
```

输出如下：

```
Does `a` require gradients? : False
Does `b` require gradients?: True
```

在神经网络中，不需要计算梯度的元素通常是叫做冻结参数。如果在你的模型中部分使用“freeze”那么你将不需要在这些参数上计算梯度（由于少计算了梯度所以客观上提升了性能）。

其他的一个使用情况就是在微调一个预先设计好的神经网络的时候冻结参数是很有必要的。

在整合过程中，我们冻结模型的大部分参数，仅仅调整分类层去预测新的标签。让我们用一个小的例子去表示这个情况。和以前一样我们导入了一个`resnet18`（残差神经网络的一种结构），然后冻结所有参数。

```python
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
```

假设我们想在一个有10个标签的新数据集上微调模型。在`resnet`中，分类器是最后的线性层`model.fc`。我们能简单地使用一个新的线性层来替代它，然后作为我们的新分类器。

```python
model.fc = nn.Linear(512, 10)
```

如今在模型中的所有参数除了`model.fc`都是被冻结的状态。仅有的参数需要计算梯度的是`model.fc`的权重和偏置。

```python
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

注意，尽管我们在优化器中注册了所有参数，但计算梯度(并因此在梯度下降中更新)的唯一参数是分类器的权重和偏差。

在`torch.no_grad()`中，也可以作为上下文管理器使用相同的排除功能。

好的烧脑的第二部分就更新完毕了，接下来还有两个部分，我会尽快更新完毕。
```
