Here is your content converted to markdown format:

```markdown
# Pytorch教程：DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ

为了大家观看方便，我在这里直接做了一个四个部分内容的跳转，大家可以自行选择观看。

- [第一部分](#第一部分)
- [第二部分](#第二部分)
- [第三部分](#第三部分)
- [第四部分](#第四部分)

Pytorch官网有非常优秀的教程，其中有几篇小短文属于名为 **DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ** 这个小专栏的内容，考虑到大家阅读英文文献有点困难，笔者打算花两天时间做一下翻译，同时结合自己的理解做一些内容调整，原文链接贴在这里[点此跳转](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)。话不多说直接进入正题。

## 第一部分：认识tensor

### Tensors

tensor直译就是张量，可以理解为向量vector的延申。tensors是一种特殊的数据结构与数组和矩阵十分相似。在Pytorch中我们使用tensors去编码一个模型的输入输出，模型的参数也是这样操作的。

tensors和Numpy的多维数组ndarray很相似，除了tensors可以运行在GPU上，或者其他的可以用来加速计算速度的特殊硬件。如果你熟悉ndarray的话，你可以正确的使用Tensors的API接口。不过如果你不熟悉的话，那请提前快速地阅读一下对应的API。

下面是加载库的头文件：

```python
import torch
import numpy as np
```

### Tensor的初始化

tensor可以使用多种方式初始化，下面是一些例子。

#### 1. 直接来源于数据

tensor可以直接使用数据创建，数据类型可以自动的识别出来。

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

#### 2. 从numpy数组中来

tensor可以使用numpy数组来创建（numpy数组就类似于常用数值型的多维数组），反之亦然。

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

#### 3. 来源于tensor

新的tensor保留了原来tensor的参数属性，包括（形状，数据类型），这里的形状可以理解为tensor的维度。

```python
x_ones = torch.ones_like(x_data)  # 保留了x_data的属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖了x_data的数据类型
print(f"Random Tensor: \n {x_rand} \n")
```

对应的输出如下：

```
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.5715, 0.8526],
        [0.2244, 0.4860]])
```

#### 4. 使用随机值或者常数值构建

shape是一个tensor的维度，在下图的函数中，它决定了输出tensor的维度。

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

输出结果如下：

```
Random Tensor:
 tensor([[0.5585, 0.0448, 0.1216],
        [0.7801, 0.0683, 0.2854]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

以上就是tensor的四种创建方式了。

### tensor的属性

tensor的属性主要包括shape（形状表示维度）、数据类型和它们存储在哪一个设备上。

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

输出如下：

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### 对Tensor的操作方式

总共大约100种对tensor的操作，包括转置，索引，切片，数学运算，线性代数，随机抽样，更多的操作描述可以点击这个[链接](https://pytorch.org/docs/stable/tensors.html)。

这些操作中的每一个都可以运行在GPU上，简单来说就是可以获得一些硬件加速。如果你财力雄厚的话可以考虑搞一块牛x的GPU帮助你干活。

#### 如果财力雄厚的话，我们可以将tensor移到GPU上

```python
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```

可以从操作清单中找出一些试试看，如果熟悉Numpy API，你会发现tensor的API很容易学会。（PS：我要是熟悉Numpy我还看你这个教程吗≧ ﹏ ≦）

#### 1. 标准的Numpy结构式的索引和切片：

```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
```

输出如下：

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

#### 2. 连接tensor

你可以使用`tensor.cat`指定给定的维度去连接一系列的tensor。

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

输出如下：

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

#### 3. tensor的乘积

```python
# 计算tensor的乘积
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# 另一种表述形式:
print(f"tensor * tensor \n {tensor * tensor}")
```

输出如下：

```
tensor.mul(tensor)
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

计算在两个tensor之间矩阵的乘积。

```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# 另一种表述形式:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

结果如下：

```
tensor.matmul(tensor.T)
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

tensor @ tensor.T
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

#### 4. 原地操作

在这些操作中有一个`_`后缀是原地操作，比如：`x_copy_(y)`，`x.t_()`，这些操作会改变x的数值。

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

结果如下：

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

值得注意的是原地操作不会保留某些中间变量，这就导致在某些需要中间量的操作中，由于中间量没有被保存会出现很多问题，因此这种写法要谨慎使用。

### 使用Numpy过渡

tensor在cpu上以及numpy数组可以共享它们潜在的内存位置，改变一个就可以改变另一个。

#### Tensor转为numpy数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

结果如下：

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

tensor中的改变反映在了numpy数组之中。

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

输出如下：

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

#### numpy数组转化为Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

numpy数组的改变体现在了tensor中。

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

输出如下：

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

我们的第一篇就到这里了，后面还会对内容进行完善，接下来会陆续更新下面几篇。
```
