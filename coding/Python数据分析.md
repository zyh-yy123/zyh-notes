# NumPy

> 简介:
>
> NumPy 是 Python 进行高效数值计算和数据分析的核心库，它的核心是多维数组对象（ndarray），以及围绕 ndarray 提供的一系列数组创建、索引与切片、通用函数（ufunc）、广播机制（broadcasting）和聚合函数（aggregation）等操作。通过这些机制，用户能够在 C 语言层面上对数据进行向量化运算，避免 Python 层面的循环开销，从而大幅提升性能



## 1 ndarray

- ndarray 是一个具有固定数据类型和固定大小的多维数组容器，元素类型由 dtype 决定；其形状（shape）由一个 N 元组定义，每个维度的大小由该元组的对应元素指定

- ndarray 对象提供了高效的内存布局，允许在 C 语言层面进行快速访问和运算，无需 Python 循环

## 2 数组创建

### 2.1 从Python列表创建

```python
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4]])
```

### 2.2 常用快捷函数

- `np.zeros(shape)`：创建全零数组
- `np.ones(shape)`：创建全一数组
- `np.arange(start, stop, step)`：类似 Python 内置 `range`，返回一维等差数组
- `np.linspace(start, stop, num)`：返回指定个数的等间隔一维数组
- `np.random` 模块：生成随机数组（如 `np.random.rand`、`np.random.randn` 

### 2.3 数组数据类型

| 数据类型（dtype） | Python 类型对应 | 简写 | 描述                          |
| ----------------- | --------------- | ---- | ----------------------------- |
| `int8`            | `int`           | `i1` | 8位整数（-128 到 127）        |
| `int16`           | `int`           | `i2` | 16位整数（-32,768 到 32,767） |
| `int32`           | `int`           | `i4` | 32位整数（常用整数类型）      |
| `int64`           | `int`           | `i8` | 64位整数（适合大整数）        |
| `uint8`           | 无符号 `int`    | `u1` | 8位无符号整数（0 到 255）     |
| `float16`         | `float`         | `f2` | 半精度浮点数（节省内存）      |
| `float32`         | `float`         | `f4` | 单精度浮点数（较节省内存）    |
| `float64`         | `float`         | `f8` | 双精度浮点数（默认浮点类型）  |
| `bool_`           | `bool`          | `?`  | 布尔类型（True/False）        |
| `str_`            | `str`           | `U`  | Unicode 字符串（定长）        |
| `object_`         | 任意对象        | `O`  | Python 对象（杂项混合数据用） |

数据类型转换：

`astype`方法
`arr1.astype(np.float64)` 表示转化为浮点型

### 2.4 创建数组副本和视图

`copy`创建一个副本

### 2.5 数组合并

`append(a,b,axis)`

首先二者维数必须一样

- axis = 0,按照行合并，二者需要有相同的列数
- axis = 1,按照列合并，二者需要有相同的行数

注意：如果不指定axis，`append`会把数组扁平化后拼接！

`concatenate((a,b),a axis)`

### 2.6 数组拼接

`hstack` horizontal stack(水平拼接) axis = 1

`vstack` vertical stack axis = 0 

只堆叠矩阵或者只堆叠向量，都可以正常工作，涉及到一维数组和矩阵的hstack会出错（尺寸不匹配）





## 3 索引和切片

### 3.1 基本切片

与 Python 列表类似，使用 `[start:end:step]` 进行切片；省略 `start`、`end` 或 `step` 分别默认为 0、维度长度、1
>注意是左闭右开

#### 一维数组

切片赋值：把一个标量赋值给一个切片时，该值回自动传播到整个选区
>```python
>arr = np.ones((10,))#生成一个一维的10个元素的数组
>arr[2,4] = 100
>#结果是整个选区都变成100
>```

注意：**数组切片是原始数组的view而不是copy，这意味着它们共享底层存储**

##### 高维数组

高维数组中，如果省略后面的索引，则会返回低一个纬度的ndarry

通过将整数索引和切片混合，可以得到低纬度切片

```python
arr = np.array([1,2,3],[4,5,6],[7,8,9])
arr[1,:2]#切出第二行前两列
arr[:2,2]#切出第三列前两行
```



### 3.2 多维索引

- **纯整型索引**：使用多个逗号分隔的索引，依次对应各个维度。

- **切片与整型混合**：如 `b[0, :]` 表示第一行所有元素。

- **布尔索引**：通过布尔型数组筛选，如 `a[a > 2]` 选出大于 2 的元素
	例如`arr[names = 'zyh']`

- `where()`可以根据条件返回数组中的值的索引，传条件进去，返回满足条件的元素的列表
	例：

	```python
	a = np.range(0,100,10)
	b = where(a<50)
	#返回 b[0,1,2,3,4]
	a[b] = [0,10,20,30,40]
	```

	

## 4 通用函数与向量化

### 4.1 算数运算

大小相等的数组之间的任何运算都会将运算应用到元素级

```python
data = np.array([1, 2, 3])
ones = np.ones(3)
data + ones    # array([2,3,4])
data * data    # array([1,4,9])
data / data    # array([1.,1.,1.])
```

均为向量化实现



### 4.2 常见函数

- `np.add`, `np.subtract`（从第一个数组中减去第二个数组中的元素）, `np.multiply`（数组元素相乘）, `np.divide（除法）`
- `np.sin`, `np.exp`, `np.log` 等数学函数
- `np.maximum`, `np.minimum` 等比较函数

### 4.3 字符串操作

分割倍乘：

`numpy.char.multiply(arr,num)`将字符串arr重复num次

字符串分割

`numpy.char.split(arr,sep='')`将字符串arr用分隔符切开（默认空格）

常用函数

| 函数名                 | 作用描述                      | 示例                                |
| ---------------------- | ----------------------------- | ----------------------------------- |
| `np.char.lower()`      | 全部小写                      | `['hello'] → ['hello']`             |
| `np.char.upper()`      | 全部大写                      | `['hello'] → ['HELLO']`             |
| `np.char.capitalize()` | 首字母大写，其余小写          | `['hello'] → ['Hello']`             |
| `np.char.title()`      | 所有单词首字母大写            | `['hello world'] → ['Hello World']` |
| `np.char.strip()`      | 去除前后空格                  | `['  hi '] → ['hi']`                |
| `np.char.lstrip()`     | 去除左侧空格                  | `['  hi'] → ['hi']`                 |
| `np.char.rstrip()`     | 去除右侧空格                  | `['hi  '] → ['hi']`                 |
| `np.char.add()`        | 字符串拼接                    | `['a'] + ['b'] → ['ab']`            |
| `np.char.multiply()`   | 重复字符串                    | `['hi'] * 2 → ['hihi']`             |
| `np.char.center()`     | 居中填充                      | `center('hi', 5, '*')` → `'*hi*'`   |
| `np.char.replace()`    | 替换子串                      | `replace('cat', 'c', 'b') → 'bat'`  |
| `np.char.find()`       | 查找子串位置（找不到返回 -1） | `find('abc', 'b') → 1`              |
| `np.char.count()`      | 子串出现次数                  | `count('banana', 'a') → 3`          |
| `np.char.equal()`      | 元素逐个字符串比较（相等）    | `['a'] == ['a'] → [True]`           |
| `np.char.not_equal()`  | 元素逐个字符串比较（不相等）  | `['a'] != ['b'] → [True]`           |
| `np.char.isdigit()`    | 判断是否全是数字字符          | `['123', 'abc'] → [True, False]`    |
| `np.char.isalpha()`    | 判断是否全是字母字符          | `['abc', '123'] → [True, False]`    |

### 4.4 数组排序

`np.sort(A,axis)`按照指定维度排序(复制一个新数组)

`np.argsort(A,axis)`返回排序数据的索引值

```python
a = np.array([30, 10, 20])
order = np.argsort(a)

print(order)  # [1 2 0] → 表示 10 最小，其次是 20，再是 30
print(a[order])  # [10 20 30]
```



## 5 广播机制

### 5.1 概念

当两个不同形状的数组进行元素级运算时，NumPy 会按照一套规则“广播”较小数组，使得它在逻辑上扩展到与较大数组相同的形状，从而无需显式复制数据

### 5.2 规则

1. 从尾部对齐两个数组的形状；
2. 如果维度相同或其中有一个维度为 1，则兼容；否则报错；
3. 维度为 1 的数组在运算时沿该维度进行复制

```python
A = np.array([[1,2,3],[4,5,6]])
b = np.array([10,20,30])
A + b  
# array([[11,22,33],[14,25,36]])
```

## 6 聚合函数

### 6.1 常用统计函数

- `np.sum`, `np.mean`, `np.std`, `np.var`
- `np.min`, `np.max`, `np.median`, `np.quantile`
- `np.cumsum`, `np.cumprod` 等累积函数

### 6.2 沿轴计算

通过参数 `axis`指定沿那个纬度聚合

```python
M = np.array([[1,2,3],[4,5,6]])
M.sum(axis=0)  # array([5,7,9])，按列求和
M.sum(axis=1)  # array([6,15])，按行求和
```

## 7 形状操作

### 7.1 重塑

#### 数组的形状

`arr.shape` 是指一个表示数组各个维度大小的元组

例如下面是一个三维数组，表示有 3 个“矩阵”，每个矩阵是 4 行 2 列

```python
c = np.ones((3, 4, 2))
print(c.shape)  # (3, 4, 2)
```

#### shape和axis

- `shape = (3, 2)`：这是一个二维数组，有 **3 行 2 列**。
- `axis`：是 NumPy 操作中的“**维度编号**”，从 **0 开始计数**。

但这里面有个**极易混淆**的问题：

> **维度编号 axis 不是“横着/竖着”的字面意思，而是对应 shape 元组中每一项的下标！**

#### reshape的使用

```python
a = np.arange(12)
b = a.reshape((3,4))  # 变为 3 行 4 列
```

#### resize

- 原地修改，而不返回一个新的数组
- 如果新数据域原数据个数不同，需要增加参数 `refcheck = False` 放弃多余数据或用0补齐不够的数据

```python
a = np.arr([1,2],[3,4],[5,6])
a.resize(2,2,refcheck = False)
#变成([1,2][3,4])
```



### 7.2 连接与拆分

- **连接**：`np.concatenate`, `np.vstack`, `np.hstack`
- **拆分**：`np.split`, `np.vsplit`, `np.hsplit`

```python
x = np.array([1,2,3])
y = np.array([4,5,6])
z = np.concatenate([x,y])  # array([1,2,3,4,5,6])
```

## 8 线性代数

### 8.1 常用功能

#### 矩阵乘法

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# 方法一：np.dot()
print(np.dot(A, B))

# 方法二：使用 @（推荐）
print(A @ B)
```

注意：**@是线性代数意义下的矩阵乘法**

#### 解线性方程组$AX=b$

```python
A = np.array([[2, 1],
              [1, 3]])

b = np.array([8, 13])

x = np.linalg.solve(A, b)
print(x)  # 解 x 的值
```

#### 求矩阵的逆

```python
A = np.array([[4, 7],
              [2, 6]])

invA = np.linalg.inv(A)
print(invA)
```

#### 特征值和特征向量

```python
A = np.array([[2, 0],
              [0, 3]])

vals, vecs = np.linalg.eig(A)
print("特征值：", vals)
print("特征向量：", vecs)
```

#### 矩阵转置

`A.T`

## 9 如何用numpy表示数学公式

| 数学符号               | NumPy 表达式               | 说明                        |
| ---------------------- | -------------------------- | --------------------------- |
| $a^T$                  | `a.T`                      | 向量/矩阵转置               |
| $a \cdot b$            | `np.dot(a, b)` 或 `a @ b`  | 点乘 / 矩阵乘法             |
| $a \times b$（叉乘）   | `np.cross(a, b)`           | 三维向量的叉乘              |
| $\|a\|$                | `np.linalg.norm(a)`        | 向量的范数                  |
| $\sum x_i$             | `np.sum(x)`                | 数组元素求和                |
| $\frac{1}{n} \sum x_i$ | `np.mean(x)`               | 平均值                      |
| $A^{-1}$               | `np.linalg.inv(A)`         | 矩阵求逆                    |
| $A^T A$                | `A.T @ A`                  | 矩阵乘法                    |
| $Ax = b$               | `np.linalg.solve(A, b)`    | 解线性方程组                |
| $\text{diag}(x)$       | `np.diag(x)`               | 以向量 x 为对角线的对角矩阵 |
| $\text{trace}(A)$      | `np.trace(A)`              | 主对角线之和                |
| $\text{rank}(A)$       | `np.linalg.matrix_rank(A)` | 求秩                        |
| $\text{det}(A)$        | `np.linalg.det(A)`         | 行列式                      |

## 10 实践

### 花式索引

数组索引可以用数组！

### 众数

`from scipy.stats import mode`

`mode(array)`返回两个值

> `mode.mode`：最常出现的数（可能是1.5、4.5等等）`mode.count`：这个数出现了几次
>
> `keepdims=False`：让返回值是**标量**，不加这个参数，默认返回的是数组。

```python
petal_len = iris_data[:,2]
mostcommon_mode = mode(petal_len,keepdims=False)
print(f"最常见的花瓣长度是:{mostcommon_mode.mode:.1f}cm")
```

### 例子

要求：

1. 根据sepallength列对数据集进行排序
2. 在鸢尾属植物数据集中找到最常见的花瓣长度值
3. 在鸢尾属数据集的petalwidth中查找第一次出现的值大于1.0的位置

```python
import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import mode
# 加载数据
iris = load_iris()

# 特征数据（150行 4列）
X = iris.data       # numpy array, shape=(150, 4)

# 类别标签（整数 0,1,2）
y = iris.target     # numpy array, shape=(150,)

# 类别名称（字符串）
target_names = iris.target_names  # array(['setosa', 'versicolor', 'virginica'])

# 如果你想把标签名替代 0/1/2
y_named = target_names[y]  # shape=(150,), 每个元素是字符串

# 最终拼接（每一行是特征 + 类别名）
iris_data = np.column_stack((X, y_named))

sepal_len = iris_data[:,0].astype(float)#把字符串转为浮点型还没法排序因为不是整数
sort_indices = np.argsort(sepal_len)#拿到长度排序的序号，这是关键
sorted_data = iris_data[sort_indices]
#采用花式索引完成按照第一列长度对数据集排序
print(sorted_data[:5])

petal_len = iris_data[:,2].astype(float)
mostcommon_mode = mode(petal_len,keepdims=False)
print(f"最常见的花瓣长度是:{mostcommon_mode.mode:.1f}cm")

petalwidth = iris_data[:,3].astype(float)
for i in range(150):
    if petalwidth[i] > 1.0 :
        print("第一次大于1.0的位置是第",i+1,"个数据")
        break

```

---

# Pandas

## 1 简介

基于 NumPy，提供了更高级的索引与分析工具，使大规模数据处理如虎添翼

核心数据结构：

- **Series**：一维数组，可存储任意数据类型，有索引标签，类似 Python 字典和 NumPy 一维数组的结合体

- **DataFrame**：二维表格型数据结构，行列均有==标签==，对齐功能强大，等同于 Excel 表、SQL 表或 R 的 data.frame，是最常用的pandas对象

## 2 Series

### 2.1 series的结构

1. **values**：一维的数据内容（如数值、字符串、布尔值等）
2. **index**：每个数据点的标签（默认是 0, 1, 2…）

### 2.2 创建series

#### 从列表创建

```python
s = pd.Series([1, 2, 3, 4])
print(s)
```

#### 指定索引创建

```Python
s = pd.Series([90, 80, 70], index=['Alice', 'Bob', 'Charlie'])
print(s)
```

类似字典，很适合做映射类的数据

#### 从字典创建(推荐)

```python
data = {'a': 100, 'b': 200, 'c': 300}
s = pd.Series(data)
```

#### 标量+index创建常数序列

```python
s = pd.Series(5, index=['a', 'b', 'c'])
```

#### 用numpy创建

```Python
import numpy as np
arr = np.array([3, 6, 9])
s = pd.Series(arr)
print(s)
```

### 2.3 series的基本属性

**使用numpy函数或运算都会保留索引值的链接**

| 属性 / 方法 | 功能                       |
| ----------- | -------------------------- |
| `s.index`   | 获取索引列表               |
| `s.values`  | 获取数值列表（NumPy 数组） |
| `s.dtype`   | 数据类型                   |
| `s.shape`   | 元素数量（元组）           |
| `s.name`    | 名称属性，可用于命名列     |

访问：

1. 索引访问，类似字典，用index访问value
2. 下标访问，类似数组，data[0]



可以将series看作一个定长的有序字典

### 2.4 series的常用方法

#### 索引

只传入一个字典，则series中的索引就是原字典的key

可以混入排好序的字典的key来改变顺序并添加或修改索引



就地修改

直接给index重新赋值

```python
a.index = ['newindex1','ni2','ni3','ni4']
```

#### 检查缺失值

`isnull()`为空输出 `True`

#### 增删

添加：`append()`

```python
a.append(pd.Series['happy','sunny'],index=['mood','weather'])
```

删除：`drop()`

类似字典，删除索引即可

#### 查看数据

`head()`返回前n行数据，如果不指定行数，默认返回前5行

`tail()`返回后5行



## 3 DataFrame

### 3.1 概述

>表格型的数据结构，含有一组有序的列，每列可以是不同的数据类型
>
>类似SQL里的表，但更厉害

每一行都可以看作一个series结构（类比SQL里的一个元组）

### 3.2 DataFrame的创建

创建空dataframe

`df = pd.DataFrame()`



导入列表或numpy数组组成的字典

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(df)
```



嵌套列表





嵌套字典

### 3.3 pandas文件读取

| 文件类型           | pandas读取函数            | 比喻                           |
| ------------------ | ------------------------- | ------------------------------ |
| `.csv`（逗号分隔） | `read_csv()`              | 通用快餐：标准、轻量           |
| `.xlsx`（Excel）   | `read_excel()`            | 精致套餐：表格 + 样式          |
| `.txt`（文本）     | `read_csv(..., sep='\t')` | 咸鱼干：结构也行但得配合调味料 |
| `.json`            | `read_json()`             | 数据界的“积木”：结构化但多维   |
| 数据库             | `read_sql()`              | 正餐食堂：要钥匙（连接）才能吃 |
| HTML 表格          | `read_html()`             | 网页抓饭：得会翻译 HTML        |

#### 3.4 pandas文件写入

`.to_格式()` 

例如：

```python
df.to_json('iris.json', orient='records', lines=True)
```

最常使用的`index=False` 的意思是：**导出文件时，不保存 DataFrame 的行索引（行号）**，只保存你看到的纯净数据表内容

#### dataframe简单操作

##### 转置

`df.T`

##### 查看维度

`df.shape` 查看形状

`df.shape[0]` 查看行数

`df.shape[1]`查看列数

`df.size` 查看尺寸

`df.column` 查看列名

`df.index` 查看行名

##### 指定列排序

指定index的顺序

##### 唯一值函数

| 函数名                  | 功能                           | 返回类型             |
| ----------------------- | ------------------------------ | -------------------- |
| `Series.unique()`       | 返回所有不重复的值（顺序保留） | `ndarray`            |
| `Series.nunique()`      | 返回唯一值的个数               | 整数                 |
| `Series.value_counts()` | 返回每个唯一值及其频数         | `Series`（自动排序） |

如果想要观察多个列组合的唯一值，可以用 `drop_duplicates()`

 默认行为：

- 比较**整行是否重复**，完全相同才删除
- 保留第一次出现的记录
- 返回新的 DataFrame，原来的不变



如果只根据某一列去重，增加关键字 `subset=`

例如 `df.drop_duplicates(subset='name')`



对于重复的几行，关键字 `keep` ，默认保留第一行

如果 `keep='last'`保留最后一次出现的行

`keep = False`表示删除所有重复组合所在的行

##### 替换





