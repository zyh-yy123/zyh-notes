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

### 3.4 pandas文件写入

`.to_格式()` 

例如：

```python
df.to_json('iris.json', orient='records', lines=True)
```

最常使用的`index=False` 的意思是：**导出文件时，不保存 DataFrame 的行索引（行号）**，只保存你看到的纯净数据表内容

###  3.5 dataframe简单操作

#### 转置

`df.T`

#### 查看维度

`df.shape` 查看形状

`df.shape[0]` 查看行数

`df.shape[1]`查看列数

`df.size` 查看尺寸

`df.column` 查看列名

`df.index` 查看行名

#### 指定列排序

指定index的顺序

#### 唯一值函数

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

#### 替换

`df.replace()`

##### 单值替换

```python
df.replace('old_value', 'new_value')
```

##### 多值替换

```python
df.replace({'A':'X','B':Y})
```

##### 指定某一列替换

```python
df['column_name'].replace('A', 'B', inplace=True)
```

例如：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'sex': ['male', 'female', 'male', 'female'],
    'age': [25, 30, 22, 28]
})

# 替换字符串
df['sex'] = df['sex'].replace({'male': 1, 'female': 0})
```

##### 逻辑替换

`where(cond)`：保留满足条件的值，**不满足的替换掉**

`mask(cond)`：**满足条件的替换掉**，不满足的保留

##### 数值替换

`round()`按精度四舍五入

`abs()`取绝对值

`clip(a,b)` 按照下界a，和上界b截断数据



### 3.6 常用基本函数

#### 头尾

`head()` 返回前n行

`tail()` 返回后n行默认值为5

#### 基本信息

`info()`

`describe()`

#### unique()

`unique()` `nunique()` 得到其唯一值组成的列表和唯一值的个数

类似 `value_count()` 得到唯一值和其对应的频数

#### 排序函数

值排序：`sort_values()` 

```python
df1.sort_values(['score','sno'],asceding=[True,False])
#表示对df按照分数升序排序，分数相同情况下按照学号降序
```

索引排序：`sort_index()`

#### apply方法

`apply()` 是 Pandas 中用于对 **Series 或 DataFrame** 的每一列或每一行应用一个**自定义函数**的利器

使用场景：

- 你想对一列数据进行 **批量处理**
- 想对某行或某列进行 **逻辑判断**、**复杂转换**
- 想使用 **自定义函数**，而不是像 `sum()`、`mean()` 这样固定的函数

#### 窗口对象

滑动窗口 `rolling`

```python
s = pd.Series([1, 2, 3, 4, 5])

s.rolling(window=3).mean()
#输出：
0    NaN
1    NaN
2    2.0   # (1+2+3)/3
3    3.0   # (2+3+4)/3
4    4.0   # (3+4+5)/3

```

扩张窗口 `expanding`

从头开始，越滚越大



指数加权窗口 `ewm`

越近的数据权重越高，用来“平滑”噪声数据

它不像 `rolling` 那样完全丢掉早期数据，而是对数据衰减加权

> 比如用于股价、传感器数据、噪声去除

`shift(n)` 取向前第n个元素的值

`diff(n) `与向前第n个元素作差

`pct_change(n) `与向前第n个元素计算增长率

#### 索引器

列索引是最常见的索引，通过 [] 实现 如 `df['Name'].head()`拿出姓名列的前五行



##### loc索引器

基本语法：

```python
df.loc[行索引, 列索引]

#行索引 可以是标签、切片、布尔列表

#列索引 同样可以是列名、切片、布尔列表

#注意：loc 的切片是“闭区间”，即包头包尾
```

几种情况：

1. 索引为单个元素
	此时直接取出相应的行或列

2. 索引为元素列表
	取出列表中所有元素值对应的行或列

3. 索引为切片
	包含两个端点，不唯一时报错

4. 索引为布尔列表

	``` python
	df.loc[df.score>60]
	#选出分数大于60的行
	```

5. 索引为函数
	这里的函数的返回值必须是上述4种之一，函数的输入值为dataframe本身
	用于写一个复杂条件

##### iloc索引器

使用与loc类似，只是对位置进行筛选

合法对象：

- 整数
- 整数列表
- 整数切片
- 布尔列表
- 函数

注意：证书切片左闭右开



##### query方法

支持把字符串形式的查询表达式传入query方法来查询数据，表达的执行结果必须返回布尔列表

>`query()` 方法允许你用**类似 SQL 的语法**，对 DataFrame 进行**行筛选**，可读性高，尤其适合做复杂筛选逻辑

比如：

```python
df.query("age > 18 and score < 90")
```

##### 索引的设置与重置

`set_index()` 主要参数 `append` 表示是否保留原来的索引，直接把新设定的添加到原索引的内层

`reindex(index=['','',''],column=['','',''])`

### 3.7 随机抽样和多级索引

#### 随机抽样

把dataframe的每一行看做一个样本，每一列看做一个特征，将整个dataframe看做整体，想要对样本或者特征进行随机抽样就可以用 `sample` 函数

```python
df.sample(n=None, frac=None, replace=False, random_state=None, axis=0)
```

| 参数名         | 说明                                       |
| -------------- | ------------------------------------------ |
| `n`            | 指定抽样的“数量”                           |
| `frac`         | 指定抽样的“比例”，如 `frac=0.1` 表示抽 10% |
| `replace`      | 是否有放回抽样，默认为 False（无放回）     |
| `random_state` | 随机种子，确保每次抽样结果一致             |
| `axis`         | 0 表示按“行”抽样（默认），1 表示按“列”抽样 |

#### 多级索引

组合键（候选键的感觉）

```python
import pandas as pd

data = {
    'year': [2023, 2023, 2023, 2024, 2024],
    'term': ['spring', 'spring', 'fall', 'spring', 'fall'],
    'name': ['Alice', 'Bob', 'Alice', 'Alice', 'Bob'],
    'score': [85, 88, 90, 92, 91]
}

df = pd.DataFrame(data)
df_multi = df.set_index(['year', 'term'])
```

多级索引表的结构：

```apl
               name  score
year term                
2023 spring  Alice     85
     spring    Bob     88
     fall    Alice     90
2024 spring  Alice     92
     fall      Bob     91
```

- `year` 是第一级索引（最外层）
- `term` 是第二级索引（内层）
- 表格左侧的行索引现在是一个**树状结构**
- 每个 `(year, term)` 对应若干条记录

---

## 4 数据处理

### 4.1 分组

```python
df.groupby(分组依据)[数据来源].使用操作
```

```python
#按照专业统计工资的中位数
df.groupby('dept')['salary'].median()
```

如果需要根据多个维度分组，只需要在分组依据中传入多个列即可

如：

```python
df.groupby(['dept','gender'])['salary'].median()
#增加一个性别分组
```

如果需要复杂条件，则需先写出分组条件

```python
conditon =  df.salary>df.salary.median()
df.groupby(condion)['salary'].mean()
```

#### groupby 对象

> `groupby` 是对数据按某些规则**分组**，再对每组数据**进行计算或操作**的过程，本质就是：
>
> **拆分 → 应用 → 合并（Split → Apply → Combine）**

```python
grouped = df.groupby('dept')
```

此时并没有真正计算！你得到的是一个 **groupby 对象**，就像“分好组但还没评分的比赛”

可以理解为：

- `grouped` 是一个“准备好”的分组对象
- 你需要“调用聚合函数”来让它动起来

#### groups属性

`groupby.groups` 是一个 **Python 字典**，键是分组的 key（分组依据的值）**，**值是这些组中对应的行索引的列表

#### 分组的三大操作

##### 聚合

虽然在groupby对象上定义了很多函数，但是依旧不方便：

- 无法同时使用多个函数
- 无法对特定列使用特定的聚合函数
- 无法使用自定义的聚合函数
- 无法直接对结果的列名在聚合前自定义命名

解决方法 `agg方法`

###### agg方法

`agg()` 是用来对分组后的数据**应用一个或多个聚合函数**，可以对不同的列使用不同的函数

```python
df.groupby('class')['score'].agg(['mean', 'max', 'min'])
```

可以理解为：

- `.agg()` = 用各种“总结性函数”把一堆数据压缩成你要的“汇总结果”
- 可以灵活地指定：
	- 单个函数名（字符串）
	- 多个函数名（列表）
	- 自定义函数（甚至 `lambda`）

可以实现对特定的列使用特定的聚合函数,传字典

```python
grouped.agg({'salary';['mean','max']},'score':'count')
```

###### 聚合结果重命名

在函数名前用新的名字

##### 连接

基本语法

```python
pd.merge(left,right,how = 'inner',on = '列名')
```

| 参数                  | 含义                                                         |
| --------------------- | ------------------------------------------------------------ |
| `left`                | 左边的 DataFrame                                             |
| `right`               | 右边的 DataFrame                                             |
| `how`                 | 合并方式：`inner`内连接、`left`保留左侧的悬浮元组、`right`、`outer`外连接，两侧悬浮元组均保留 |
| `on`                  | 依据哪些列进行连接                                           |
| `left_on`、`right_on` | 如果左右列名不同，用这两个参数                               |

###### 索引连接

`join` 默认是索引连接

```python
pd.join(df2, how='inner')  
```

###### 方向连接

用户不关心怎么合并，只希望拼接类似SQL中的 `UNION`

`concat`

```python
pd.concat(objs, axis=0, join='outer', ignore_index=False)
```

| 参数           | 说明                                                     |
| -------------- | -------------------------------------------------------- |
| `objs`         | 要拼接的对象列表，如 `[df1, df2]`                        |
| `axis`         | 拼接方向，`0` 代表行拼接（垂直），`1` 代表列拼接（水平） |
| `join`         | `outer`（并集）、`inner`（交集），决定对齐方式           |
| `ignore_index` | 是否重置索引为默认整数索引                               |

```python
pd.concat([df1,df2],axis = 1)
```

默认情况下 join = outer ，即保留所有悬浮元组

join = inner 内连接

##### 变形

###### 长宽表的变形

- **宽表（Wide format）**：每一个变量占一列，行是观察对象。
- **长表（Long format）**：每一行是一个观测值，列通常是：id、变量名、变量值。

例如：上面是宽表，下面是长表

>| 学号 | 姓名  | 数学 | 语文 | 英语 |
>| ---- | ----- | ---- | ---- | ---- |
>| 1    | Alice | 90   | 85   | 95   |
>| 2    | Bob   | 88   | 82   | 91   |
>
>| 学号 | 姓名  | 科目 | 成绩 |
>| ---- | ----- | ---- | ---- |
>| 1    | Alice | 数学 | 90   |
>| 1    | Alice | 语文 | 85   |
>| 1    | Alice | 英语 | 95   |
>| 2    | Bob   | 数学 | 88   |
>| 2    | Bob   | 语文 | 82   |
>| 2    | Bob   | 英语 | 91   |

`pd.pivot()`

长表变宽表的函数

```python
df_wide = df_long.pivot(index=['学号', '姓名'], columns='科目', values='成绩')
df_wide = df_wide.reset_index()  # 恢复为普通列
index: 哪些列作为行索引

columns: 哪一列变成列名

values: 哪一列作为值填入表中
```

>`index` 相当于设置主键，`columns` 设置新的列，`values`是列对应的值

pivot的使用依赖唯一性条件，不满足时，必须通过聚合操作使得相同行列组合对应的值变成一个值

`pivot_table()`

```python
pd.pivot_table(
    data, 
    values=None, 
    index=None, 
    columns=None, 
    aggfunc='mean', 
    fill_value=None,
    margins=False
)
```

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| `data`       | DataFrame                                                    |
| `values`     | 哪一列（或多列）需要聚合                                     |
| `index`      | 透视表的行标签（类似 SQL 的 `GROUP BY`）                     |
| `columns`    | 透视表的列标签（横向展开）                                   |
| `aggfunc`    | 聚合函数，如 `'mean'`, `'sum'`, `'count'`, `np.max`, 自定义函数等 |
| `fill_value` | 用于填充空值                                                 |
| `margins`    | 是否添加总计行/列（"All"）                                   |

例子：

```python
df = pd.DataFrame({
    '姓名': ['Alice', 'Alice', 'Bob', 'Bob', 'Bob'],
    '科目': ['数学', '英语', '数学', '英语', '语文'],
    '成绩': [90, 95, 88, 92, 85]
})

pd.pivot_table(df, values='成绩', index='姓名', columns='科目', aggfunc='mean')
#输出：
科目    语文   数学   英语
姓名                   
Alice  NaN  90.0  95.0
Bob    85.0 88.0  92.0
```

`pd.melt()`

宽变长

```python
df_long = pd.melt(df, id_vars=['学号', '姓名'], var_name='科目', value_name='成绩')
```

- `id_vars`: 保留的列
- `var_name`: 原来宽表里的列名变成这里的“变量名”
- `value_name`: 宽表中各列的值，变成一个“变量值”列

注意，二者都可生成多级索引

###### 索引的变形

`unstack` 把行索引转为列索引

`stack` 列索引转为行索引

### 4.2 文本数据

#### str对象

> Pandas 的 `.str` 是 **矢量化字符串方法接口**，用于对 Series 中的字符串进行逐元素操作，就像 Python 的字符串函数，但作用于整个列

| 类别              | 方法                         | 示例                     | 作用                       |
| ----------------- | ---------------------------- | ------------------------ | -------------------------- |
| **大小写**        | `str.lower()`                | `df['姓名'].str.lower()` | 全部小写                   |
|                   | `str.upper()`                | `str.upper()`            | 全部大写                   |
|                   | `str.title()`                | `str.title()`            | 首字母大写                 |
| **查找定位**      | `str.contains('abc')`        | `str.contains('mail')`   | 是否包含子串（返回布尔值） |
|                   | `str.startswith('a')`        |                          | 是否以某字符开头           |
|                   | `str.endswith('.com')`       |                          | 是否以某字符结尾           |
| **替换**          | `str.replace('old', 'new')`  |                          | 字符串替换（支持正则）     |
| **分割**          | `str.split('@')`             |                          | 拆分成列表                 |
|                   | `str.split('@').str[0]`      |                          | 拆分后取前半部分           |
| **提取**          | `str.extract(r'@(.*)\.com')` |                          | 提取正则匹配子串           |
| **长度/空值处理** | `str.len()`                  |                          | 每个字符串长度             |
|                   | `str.strip()`                |                          | 去除首尾空格               |
|                   | `str.isnumeric()`            |                          | 判断是否全是数字           |

##### []索引器

在字符串中，可以用 [] 取出某个位置的元素

#### 文本数据的操作

##### 拆分

`str.split`

> `str.split()` 是 Pandas 的**矢量化字符串拆分方法**，它对整列文本**逐个字符串进行分割**，并返回一个新的 Series 或 DataFrame

```python
Series.str.split(pat=None, n=-1, expand=False)
```

| 参数     | 含义                                      |
| -------- | ----------------------------------------- |
| `pat`    | 拆分符（默认是空格）                      |
| `n`      | 拆分次数（默认 `-1` 表示全部）            |
| `expand` | 是否展开为多个列（`True` 返回 DataFrame） |

##### 合并



`str.cat()`

列间的合并

```python
df = pd.DataFrame({
    '姓': ['张', '李', '王'],
    '名': ['三', '四', '五']
})

df['姓名'] = df['姓'].str.cat(df['名'])
#输出
   姓  名   姓名
0  张  三  张三
1  李  四  李四
2  王  五  王五
#如果加分隔符
df['姓名'] = df['姓'].str.cat(df['名'], sep='·')

```

`str.join()`

字符串内的合并

```python
s = pd.Series(['ABC', 'XYZ'])
s.str.join('-')
#输出：
0    A-B-C
1    X-Y-Z
```

##### 匹配

`str.contains`

返回每个字符串是否包含正则模式的布尔序列

- `\s`匹配一个空格 
- `\w`匹配一个字母/数字/下划线

`str.find`和 `str.rfind`

分别表示从左向右和从右向左查询位置的索引，没查到返回-1

##### 替换

`str.replace()`

- \d  匹配一个0-9的数字
- \?  匹配字面意义问号
- |   逻辑或

##### 提取

`str.extract`

是 Pandas 中**最强大的字符串提取方法之一**，用于从字符串中**使用正则表达式提取子串**。它的功能远比 `.split()` 更灵活

> `.str.extract()` 用正则表达式提取字符串中你想要的**子部分**，提取的是**捕获组**（即正则中的小括号）

基本语法：

```python
Series.str.extract(pat, flags=0, expand=True)
```

| 参数     | 含义                                        |
| -------- | ------------------------------------------- |
| `pat`    | 正则表达式，必须包含**括号 ( )** 表示捕获组 |
| `expand` | 是否展开为 DataFrame（默认 `True`）         |

---

# 数据分析

## 1 预处理

### 1.1 大数据的概念

技术特点:

1. 数据量大 VOLUME
2. 数据类型多 VARIETY
3. 处理速度快 VELOCITY
4. 价值密度低 VALUE

大数据是系统性方法论

- 从数据驱动，自底向上的角度解决问题

### 1.2 数据分析的流程

#### 1.2.1 数据收集

1. 数据来源
	>1. 内部数据库
	>2. 外部API
	>3. 社交媒体
	>4. 公开数据集

2. 数据抓取
	>使用网络爬虫自动收集公开的数据，减少手动操作，提高数据收集效率
	>>网络爬虫：
	>>
	>>1. 概念：自动抓取网页信息的程序，用于大规模数据集
	>>2. 工作原理：
	>>	通过URL访问网页，解析HTML，提取所需数据，存储并跟踪链接进行深度搜索
	>>3. 类型：
	>>	通用爬虫，覆盖广泛网站
	>>	聚焦爬虫，特定主题或领域
	>>4. 法律与伦理
	>>	遵守 robts.txt 规则，尊重版权

3. 数据存储
	>- Hadoop HDFS,适用于大规模数据的分布存储
	>- 云存储服务，实现安全存储和快速访问

4. 数据清洗
	>- 取出明显出错数据，保证数据质量
	>- 清理无关信息

##### 网络爬虫

###### HTML简介

HTML的基本组成部分：

1. 标签
	>承载页面要显示的内容

2. 层叠样式表（CSS）
	>负责对页面的渲染

3. JavaScript
	>空置页面的交互式行为

HTML以标签为单位，不同标签提供不同的内容

```html
<标签名 属性1=属性值1 属性2=属性值2 ...>标签和内容</标签名>
<标签名 属性名1=属性值1 属性名2=属性值2···>
<<标签名 属性名1=属性值1 属性名2=属性值2···/>
```

```html
<h1>一级标题</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>
```

p标签表示段落

```html
<p>这是一个段落</p>
```

a标签表示超链接，使用时需要指定链接地址（由href属性来指定）和在页面上显示的文本：

```html
<a href="http://www.baidu.com">点这里</a>
```

table、tr、td标签 ——table标签用来创建表格，tr用来创建行，td用来创建单元格

```html
<table border="1">
<tr>
<td>第一行第一列</td>
<td>第一行第二列</td>
</tr>
<tr>
<td>第二行第一列</td>
<td>第二行第二列</td>
</tr>
</table>
```

ul、ol、li ——ul标签用来创建无序列表，ol标签用来创建有序列表，li标签用来创建其中的列表项

div标签用来创建一个块

###### CSS语法





#### 1.2.2 数据处理

- 数据清洗
	>去除错误、缺失、格式不正确或重复的数据，确保数据质量
	>
	>提高数据分析的准确性和可靠性

- 数据转换
	>将数据标准化，编码分类变量，便于模型理解和处理

- 数据集成
	>合并来自不同源的数据，解决数据冲突，创建统一视图

##### 检查数据

- 查看一列的一些基本统计信息：`data.columnname.describe()`
- 选择一列：`data['columnname']`
- 选择一列的前几行数据：`data['columnsname'][:n]`
- 选择多列：`data[['column1','column2']]`
- Where 条件过滤：`data[data['columnname'] >[condition]`

##### 数据质量的分析

处理：

- 缺失值
- 异常值
- 不一致的值
- 重复数据
- 带有特殊符号的数据

###### 缺失值处理

`data.isnull()` 检查缺失值

产生原因：

- 有的信息暂时无法获取，或者获取信息的代价太大 

-  有些信息是被遗漏的。可能是因为输入时认为不重要、忘记填写或对数据理解错误等一些人为因素而遗漏，也可能是由于数据采集设备的故障、存储介质的故障、传输媒体的故障等非人为原因而丢失。 

-  属性值不存在。在某些情况下，缺失值并不意味着数据有错误。对一些对象来说某些属性值是不存在的，如一个未婚者的配偶姓名、一个儿童的固定收入等

处理方法：

赋值默认值

`data.column.fillna('blablabla')`

`data.column.fillna(data.column.mean())`用平均值赋值



删除缺失行

`data.dropna()`

也可以增加一些限制或改动，比如 `data.dropna(how='all')`表示删除一整行

`data.dropna(thresh=5)`表示一行中只要有5个非空数据就保留



删除缺失率高的列

增加 `axis = 1`即可



###### 规范化数据类型

有时候字符型和数值类型的数字会被混淆，可以规范化，比如：

```python
data=pd.read_csv('',dtype={'column1':int})
```

这样就告诉了column1中是int类型



###### 必要的变换

大小写等等



###### 重复数据

```python
df.drop_duplicates(['first_name','last_name'],inplace=True)
```

###### 异常值

数值明显偏离其余观测值，离群点

- 可以先描述性统计，根据常识判断异常（年龄199岁等）
- 3σ原则
	如果数据服从正态分布，在3σ原则下，异常值被定义为一组测定值中与平均值的偏差超过3倍标准差的值。在正态分布的假设下，距离平均值3σ之外的值出现的概率为P( |x-μ|>3σ)≤0.003，属于极个别的小概率事件。如果数据不服从正态分布，也可以用远离平均值的多少倍标准差来描述
- 箱型图分析

#### 1.2.3 数据分析

1. 数据探索分析
	>1. 统计摘要了解数据基本特性
	>2. 可视化图表展示数据分布
	>3. 识别异常值和模式

2. 特征工程
	>1. 选择对模型有帮助的数据特征
	>2. 构建新的特征以增强模型表现
	>3. 进行特征数据的降维处理

3. 模型选择和评估
	>1. 对比多种机器学习算法性能
	>2. 选择最适合当前任务的模型
	>3. 使用交叉验证方法评估模型稳定性

---

## 2 数据可视化基础

