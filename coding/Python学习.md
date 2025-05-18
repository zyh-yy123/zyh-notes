# ==Python学习==



##  一、基本内容

### 1、**注释**

- **注释符号**  #
- 整型      整数  输出时不需要引号
	浮点型  小数

### 2、**变量**

- 与c++不同，不需要提前定义变量类型
- 例如 `name=zyh`
- 命名时不能用数字开头
- Python中的命名风格 **小写字母和下划线**

### 3、**基本运算**

>最好是在四则运算中添加空格

-  **乘法** *

-  **除法** /
	>计算结果一定是浮点型

-  **取整数商**
	符号 `//`  例如 `10//3` 值为3

-  **取余数** 
	符号 `%`  例如 `10%3`  值为1

### 4、**字符串操作**

- 可以直接用 + 完成拼接字符串操作，非常自由

	```python
	arr1="hello "
	char2="world !"
	char3=arr1 + char2 + "!!!"
	print(char3)
	```

- 也可以视为特殊的元组
	不可以修改
	可以通过下标访问

- 方法：
	>`index()` 查找起始下标
	>
	>`replace("1","2")` 把字符串1替换为字符串2（并非修改而是生成新的字符串）
	>
	>`split（）`  按照指定的分隔符分割字符串，存入列表对象中，字符串本身不变，而是得到一个列表对象
	>
	>`count（）` 统计对应字符串的次数

- 字符串比较大小

#### 字符串切片

>Python中的切片（Slicing）语法非常强大，允许你从序列（如字符串、列表、元组等）中提取子序列。切片的基本语法如下：
>
>```python
>sequence[start:end:step]
>```
>
>- **start**：切片开始的位置（包含该位置），默认为 0
>- **end**：切片结束的位置（不包含该位置），默认为序列的结尾 **左闭右开**
>- **step**：步长（即每次跳跃的步幅），默认为 1
>
>### 例子
>
>1. **基本切片**：
>
>	```python
>	s = "Hello, World!"
>	print(s[0:5])  # 输出 "Hello"
>	```
>
>	- `s[0:5]` 从索引 0 开始，直到索引 5（但不包括索引 5），即提取 `"Hello"`。
>
>2. **步长切片**：
>
>	```python
>	s = "Hello, World!"
>	print(s[::2])  # 输出 "Hlo ol!"
>	```
>
>	- `s[::2]` 表示从头到尾，以步长为 2 进行切片，得到 `"Hlo ol!"`
>
>3. **省略开始和结束索引**：
>
>	```python
>	s = "Hello, World!"
>	print(s[:5])   # 输出 "Hello"
>	print(s[7:])   # 输出 "World!"
>	```
>
>	- `s[:5]` 从开头到索引 5（不包括 5），即 `"Hello"`。
>	- `[7:]` 从索引 7 到字符串的结尾，得到 `"World!"`
>
>4. **反向切片**：
>
>	```python
>	s = "Hello, World!"
>	print(s[::-1])  # 输出 "!dlroW ,olleH"
>	```
>
>	- `s[::-1]` 用步长 `-1` 反向切片，得到反转的字符串。
>
>5. **带负数索引**：
>
>	```python
>	s = "Hello, World!"
>	print(s[-6:-1])  # 输出 "World"
>	```
>
>	- `s[-6:-1]` 使用负数索引，从倒数第 6 个字符开始，到倒数第 1 个字符（不包括该字符），得到 `"World"`。
>
>### 结合使用
>
>你可以灵活结合 `start`、`end` 和 `step` 来进行更复杂的切片操作。
>
>```python
>s = "Hello, World!"
>print(s[1:10:2])  # 输出 "el,W"
>```
>
>- `s[1:10:2]` 从索引 1 到索引 10，步长为 2，提取了 `"el,W"`。
>
>

### 5、**格式化输出**

- 按照指定的格式输出数据包括 控制格式精度小数位数等

- 通常使用 *占位符* 和 *格式说明符* 来指定输出的内容

-  例如 

	```python
	name="zyh"
	print(f"welcome {name} to my home")
	```

### 6、**真与假**

- **布尔数** 
	`True`
	`False`
	*注意首字母大写*

- **比较运算**

	等于 ==

	不等于 ！=
	运算的结果得到一个布尔数

- **逻辑运算**
	**与** and
	**或** or 
	**非** not

### 7、**简单判断**

- 简单 if 判断

	```python
	if True:
	    print("a")
	if False:
	    print("b")
	#归属于False的代码直接跳过
	```

- **判断条件** 
	注意优先级，建议添加括号

- **空值** 
	一个变量有数据则等价于True
	`None`  代表空值，作为判断条件时等价于False 

- **代码块**

	缩进四个空格的代码，属于同一层级

	同一个层级为一块

### 8、**稍复杂判断**

- **if else**
	else 跟最近的 if 配对

- **elif**
	就是 else if 的缩写
	if elif else 只会执行三者其一的代码块

	

### 9、**列表（数组）**

- **创建列表**
	`list_a=["a","b","c"]`
	可以直接用 `print` 打印列表
- **查找列表中的元素**
	*索引*  从0开始 list_a[0]
	~~反向索引~~ list_a[-n] 访问倒数第n个元素
- **修改列表中的元素**
	`list_a[0]="k"`
- **删除列表中的元素**
	`list_a.pop(1)` 表示删除第二个元素
	pop() 相当于一个函数
- **切片**
  `list_a[0:2]` 表示从第一个取到第二个
  *左闭右开原则*
  第一个元素不填时，默认为零
  第二个不填，默认到最后
- **追加元素**
  在末尾增加新的元素
  `list_a.append()`
  `append()` 是一个函数
  
  >*例：*给你一个数组 `nums` ，数组中有 `2n` 个元素，按 `[x1,x2,...,xn,y1,y2,...,yn]` 的格式排列。
  >
  >请你将数组按 `[x1,y1,x2,y2,...,xn,yn]` 格式重新排列，返回重排后的数组。
  >
  >```python
  >ret = []
  >for i in range( n ):
  >    ret.append(nums[i])
  >    ret.append(nums[i + n])
  >return ret
  >```
  >
  >
  >*例* 给你一个 **从 0 开始的排列** `nums`（**下标也从 0 开始**）。请你构建一个 **同样长度** 的数组 `ans` ，其中，对于每个 `i`（`0 <= i < nums.length`），都满足 `ans[i] = nums[nums[i]]` 。返回构建好的数组 `ans` 。
  >
  >**从 0 开始的排列** `nums` 是一个由 `0` 到 `nums.length - 1`（`0` 和 `nums.length - 1` 也包含在内）的不同整数组成的数组。
  >
  >```python
  >ans = []
  >for i in range(len(nums)):
  >     ans.append(nums[nums[i]])
  >return ans
  >```
  >
- **追加列表**  `extend()`
- **元素的插入**
  `insert（place,元素）` 在哪个位置插入什么元素

### 10、**循环**

- **for循环**

	例如想要输出一个数组中的元素时

	```python
	list=[1,2,1,3,4]
	for num in list:
	    print(num)
	```

	变量名自由选取

	**遍历**

	对所有元素进行访问

- **range()** 
	填入一个数字，不包含 
	用于创建一个整数列表，可用于for 循环中，想要执行几次num就为几
	比如下面用于求和1到100

	```python
	num=0
	for i in range(101):
	    num+=i
	```

	`range(a,b)`  表示从a开始到b，不包含b.0

	`range(a,b,c)` 表示从a开始到b，不包含b，并且间隔为c
	                      也就是*开始，结束，步长*

- **计数器**

	用于追踪循环已经执行了多少次
	类似累加

	```python
	count=0
	for i in range(101):
	    count+=1
	    不啦不啦不啦。。。
	```

- **while循环**
  例如

  ```python
  price = 2000
  while price < 3000:
      print(f"以{price}价格抢购")
      price += 200
  ```

- **break语句**
  终止循环

- **continue**
  跳过当前循环剩余部分，直接进入下一轮循环

### 11、**元组tuple**

- **不可改变元素**
	用 `（）`  生成，若只有一个元素则还需要添加逗号
	一般用下划线命名法 `tuple_a`,元组内可以是不同类型的数据类型

- 元组之间可以进行运算
	 `+`  连接

- 可以使用 `del` 删除整个元组 

- 一些方法
	>`index()` 查找数据，返回下标，否则报错
	>
	>`count()` 统计某个元素的数量
	>
	>
	
	

### 12、**字典**

- 类似于简单版指针
	由 **键（key）** 和 **值（value）**组成，二者一一对应

	命名方式一般用下划线命名法

	例如

	```python
	a_dic={"name":"zyh","age":"19","college":"CoME"}
	```

- 查找

	```python
	Info = a_dic["college"]
	```

- 添加元素
	添加新的键值队

	```python
	a_dic["gender"] = "male"
	```

- 修改元素

	```python
	a_dic["age"] = "20"
	```

- 删除元素
	删除索引（键）就可删除一个元素
	删除整个字典

	```python
	del a_dic["name"]
	del a_dic
	#例：
	#给你一个由小写英文字母组成的字符串 s ，请你找出并返回第一个出现 两次 的字母。注意：如果 a 的 第二次 出现比 b 的 第二次 出现在字符串中的位置更靠前，则认为字母 a 在字母 b 之前出现两次。s 包含至少一个出现两次的字母。
	has = {}
	for a in s:
	    if has.get(a):
	        return a
	    has[a] = 1  #将第一次出现的字母的值赋为1，也就是把字母加入字典，这样第二次遇到时执行return
	# 字典的get方法：retun the value for key if key is in the dictionary,else default
	```

### 13、**函数入门**

- 函数的定义

	```python
	def fun_a(s):
	    函数体
	    return 不啦不啦不啦
	```

- 函数名
	数字不能作为开头

- 函数参数

	多个参数，用逗号隔开

- 函数返回值
	向外部输出
	无返回值的函数实际上返回了None这个字面量（类似于void函数不需要写return）
	
- 函数的三种参数

	- 必选参数
		都要赋值

	- 默认参数
		不输入时，有默认值，在定义函数时赋初值

	- 不定长参数
		使用场景：输入数量不一定的参数

		```python
		def sum(*num):
		    sum_num = 0
		    for i in num:
		        sum_num += i
		    return sum_num
		```

- 作用域
	>局部变量
	>
	>全局变量
	>
	>可通过**global**将局部变量变成全局变量
	>
	>```python
	>num = 0
	>def fun():
	>    global num
	>    num = 6
	>```

### 14、**类和对象**

- 类的定义
	
	```python
	class Stu:
	    def __init__(self,gender,age):      #初始化函数，类似于构造函数
	        self.gender = "male"
	        self.age = 18
	    def print_age(self):
	        print self.age
	        
	```
	
- 类的方法（成员函数）

- 对象

	```python
	stu_a = Stu("male","20") 
	```

- 面向对象编程
	

### 15、**类和函数**

- 封装
	隐藏内部的变量和过程代码，仅对外暴露公开的借口
	公开的借口包括：函数名，类的方法名，需要传入的参数，会返回的数据
- 变量作用域
	私有变量需要在内部访问
- 跳转

---

## 二、**进阶学习**

### 1、**位运算符**

#### 1.1布尔位运算符

- 按位与 &
	参与运算的两个值，如果两个相应位都为1，则结果为1，反之0

- 按位或 |
	参与运算的两个值，如果相应位置有一个为1则结果为1

- 按位异或 ^

	参与运算的两个值，如果相应位置相异，结果为1

- 按位取反 ~
	把每个二进制位取反

	>正数的补码是其本身，负数的补码为正数数值二进制位取反后加一，符号位为1

#### 1.2移位运算符

- 左移动运算符 <<
	运算数的各二进位全部左移对应位数，高位丢弃，低位补0
- 右移动运算符 >>
	运算数的各二进位全部右移对应位数，高位丢弃，低位补0

>例：两个整数之间的 [汉明距离](https://baike.baidu.com/item/汉明距离) 指的是这两个数字对应二进制位不同的位置的数目
>
>给你两个整数 `x` 和 `y`，计算并返回它们之间的汉明距离
>
>```python
>return bin(x^y).count('1')
># 十进制数可以直接进行按位异或运算，得到结果为十进制数（内部进行了几次进制间的转化），通过bin（）再次转为二进制，再通过计数函数 count() 统计1的个数即可
>```
>
>



### 2、集合

优越性：不存在重复元素

定义： {}
`a = set{} `   定义空集合 `a = set()`

- 无序性 无法通过下标访问

- 修改方法：
	>`add()`      添加元素
	>
	>`remove()` 移除元素
	>
	>`pop()`      随机取出元素，同时集合本身被修改
	>
	>`clear()`  清空集合 

- 差集 生成一个新的集合 A-A交B
	`set_c = set_a.difference(set_b)`

- 可以通过for循环遍历

- 运算：

- | 运算符 | 方法                     | 含义                             | 示例             | 结果           |
	| ------ | ------------------------ | -------------------------------- | ---------------- | -------------- |
	| `&`    | `intersection()`         | 交集（都出现）                   | `set1 & set2`    | `{3}`          |
	| `      | `                        | `union()`                        | 并集（所有元素） | `set1          |
	| `-`    | `difference()`           | 差集（A有B没有）                 | `set1 - set2`    | `{1, 2}`       |
	| `^`    | `symmetric_difference()` | 对称差集（A或B有，但不都同时有） | `set1 ^ set2`    | `{1, 2, 4, 5}` |

#### 数据容器补充

通用统计功能：
>`len()` 统计元素个数
>
>`max()` 统计最大元素
>
>`min()` 统计最小元素

通用转化功能
>`list()`
>
>`str()`
>
>`tuple()`
>
>`set()` 
>
>传入数据容器便可转化为对应

通用排序功能
>`sorted(容器)`
>
>排序结果会得到列表



### 3、函数

#### 函数多返回值

按照返回值的顺序，写对应顺序的多个变量接受即可，支持不同类型的数据类型

#### 函数参数

- 位置参数
	调用函数时根据函数定义的参数位置来传递参数

- 关键字参数
	调用时通过“key = value”形式传递参数
	优势：清楚了参数的顺序需求
	
	```python
	def stu_info(name,age,gender):
	    print(f"名字是{name}年龄是{age}性别是{gender}")
	#位置参数
	stu_info("zyh",19,"male")
	#关键字参数
	stu_info(gender="male",age=19,name="zyh")
	    
	```
	
	

- 缺省参数
	又称默认参数，用于定义函数，为参数提供默认值，所有位置参数需要在默认参数之前

- 不定长参数
	位置传递

	```python
	def stu_iofo(*args):
	    print(args)
	stu_info("zyh",18,"male")
	#传入所有参数会被args收集，根据传进参数的位置合并为一个元组
	```


	关键字传递
	
	```python
	def stu_info(**kwargs):
	    print(kwargs)
	stu_info(name="zyh",age=19,gender="male")
	#key value对被接收组成字典
	```
	
	---

#### 函数作为参数传递

**计算逻辑的传递而非数据的传递**

---

#### lambda匿名函数

`lambda` 是 Python 中的一种匿名函数，适用于定义简单的、一次性使用的函数。它的语法如下：

```python
lambda 参数1, 参数2, ...: 表达式（一行代码）
```



1. **匿名**：`lambda` 定义的函数没有名称，适用于临时用途。
2. **单行**：只能写**一个表达式**，不能包含多条语句。
3. **返回值**：表达式的计算结果即为返回值，无需使用 `return` 关键字



```python
print((lambda x : x**2)(4))
#输出16
#若想多次使用可以赋值给变量
square = lambda x : x**2
print(square(6))
```

---

### 4、python文件操作

#### 4.1 文件编码

UTF-8最常用

#### 4.2 文件的读取

`open(name,mode,encoding)`

- name:目标文件名的字符串(绝对路径或相对路径)
- mode:设置打开文件的模式（访问模式）：只读、写入、追加……
- encoding:编码格式（推荐UTF-8）

```python
file1 = open('py.txt','r',encoding = 'UTF-8')
#此时file1是open函数的文件对象的一个实例
#encoding需要关键字传参
```



| 模式 | 描述                                                         |
| ---- | ------------------------------------------------------------ |
| r    | 只读，文件的指针放在文件的开头                               |
| w    | 写入，若文件存在则打开文件并从头开始编辑，清空原内容；文件不存在则创建 |
| a    | 在文件末尾追加内容；不存在创建新文件                         |



##### 4.2.1 读操作相关方法

`read（）`

- `file.read(num)`
  num表示 要从文件中读取的数据的长度（单位是字节），默认读取全部数据
  下一个 `read`会从上一次结尾继续



`readlines()`

- 按照行的方式把整个文件的内容一次性读取，返回一个列表，一行数据为一个元素



`readline()`

- 一次读取一行内容

for循环读取文件行

```python
for line in open("nihao.txt",'r'):
    print(line)
```



`close()`



`with open("file","r") as f` 

- with可以自动关闭



---

##### 4.2.2 文件的写入操作

1. 打开
	`f = open("file","w")`
2. 读写
	`f.write("nihao!") ` 文件此时在缓冲区类似 `git add`
	`f.flush()` 真正写入文件（类似 `git commit -m`）
3. 关闭

---

##### 4.2.3 文件的追加操作

类似，不过是在末尾追加新的内容

---



### 5、Python异常

所谓BUG

#### 5.1 捕获异常

对BUG进行提醒，整个程序继续运行

基本语法：

```python
try:
    可能发生错误的代码
except:
    如果出现异常执行的代码
#可以捕获所有异常
```

捕获特定的异常：

```python
try:
    # 可能会引发 ZeroDivisionError 或 TypeError 的代码
    result = 10 / 0
except (ZeroDivisionError, TypeError) as e:
    print("出现了除零或类型错误：", e)

```

若无异常，可写 `else`

`finally` 无论是否发生异常都会执行。它常用于清理资源，比如关闭文件、释放网络连接等

```python
try:
    f = open("example.txt", "r")
    content = f.read()
except FileNotFoundError as e:
    print("文件未找到：", e)
else:
    print("文件内容：", content)
finally:
    f.close()
    print("文件已关闭。")

```

---

#### 5.2 异常的传递

异常的传递（也称为异常的传播或异常的传播机制）指的是在 Python 中，当一个异常没有在当前函数或代码块内被处理时，它会沿着函数调用栈向上“传播”，直到找到可以处理它的 `except` 块，或者直到传播到程序的最外层。 如果没有被任何地方捕获，程序就会终止并输出错误信息

```python
def function_a():
    print("在 function_a 中")
    raise ValueError("这是来自 function_a 的异常")

def function_b():
    print("在 function_b 中")
    function_a()  # 调用 function_a，会抛出异常

def function_c():
    print("在 function_c 中")
    function_b()  # 调用 function_b，会继续传播异常

try:
    function_c()
except ValueError as e:
    print(f"捕获到异常：{e}")
```

>**`function_c`** 调用了 **`function_b`**。
>
>**`function_b`** 调用了 **`function_a`**，并在 **`function_a`** 内抛出了 `ValueError` 异常。
>
>异常没有在 **`function_a`** 内部被捕获，它会沿着调用栈向上传递，直到传递到 **`function_b`**。
>
>在 **`function_b`** 中，异常没有被处理，于是它继续向上传递，传递到 **`function_c`**。
>
>在 **`function_c`** 中，异常仍然没有被捕获，最终它被传递到外层的 `try...except` 块，成功被捕获并打印异常信息

##### 异常传播的规则

- **异常自下而上传播**：当在一个函数中抛出异常时，Python 会将其传播到调用它的地方。如果这个调用的函数没有处理异常，它会继续向上传播，直到找到处理该异常的 `except` 块。

- **调用栈的顺序**：异常会沿着调用栈依次向外传播，即先抛出的异常在栈底，最终捕获的 `except` 块在栈顶。

- **异常处理的优先级**：如果在多个地方可以捕获异常，Python 会按从内到外的顺序查找，直到找到一个匹配的 `except` 块

---

### 6、模块

模块是一个 .py 文件，能够定义函数类和变量，也包含代码

作用：
快速实现一些功能，可以理解为一个工具包

#### 6.1 模块的导入方式

常见方式：

1. **导入整个模块**

	使用 `import` 语句导入整个模块后，需要通过模块名来访问其中的函数、类或变量。这种方式有助于避免命名冲突。

	```python
	import math
	result = math.sqrt(16)
	print(result)  # 输出：4.0
	```

2. **导入模块中的特定内容**

	如果只需要模块中的某个函数、类或变量，可以使用 `from ... import ...` 语法。这种方式可以直接使用导入的内容，无需模块名前缀。

	```python
	from math import sqrt
	result = sqrt(16)
	print(result)  # 输出：4.0
	```

3. **导入模块并重命名**

	使用 `import ... as ...` 语法，可以为导入的模块指定一个别名。这在模块名较长或避免命名冲突时非常有用。

	```python
	import numpy as np
	array = np.array([1, 2, 3])
	print(array)
	```

4. **导入模块中的所有内容**

	使用 `from ... import *` 语法，可以将模块中的所有非私有成员导入到当前命名空间。这种方式可能导致命名冲突，因此不推荐在大型项目中使用。

	```python
	from math import *
	result = sqrt(16)
	print(result)  # 输出：4.0
	```

5. **使用 `importlib` 动态导入模块**

	Python 的 `importlib` 模块提供了在运行时导入模块的功能。这对于需要根据字符串名称导入模块的场景非常有用。

	```python
	import importlib
	math = importlib.import_module('math')
	result = math.sqrt(16)
	print(result)  # 输出：4.0
	```

**注意事项：**

- **命名空间管理**：使用 `import` 导入整个模块时，模块名作为命名空间，可以避免与当前命名空间中的名称冲突。使用 `from ... import ...` 导入特定内容时，需要注意可能的命名冲突。
- **导入顺序**：PEP 8 建议将导入分为三部分，按顺序排列：标准库导入、第三方库导入和本地应用/库导入，并且每部分之间用空行分隔。
- **避免使用 `from ... import \*`**：这种方式会将模块中的所有非私有成员导入到当前命名空间，可能导致命名冲突，降低代码可读性和维护性。

选择适当的导入方式，有助于提高代码的可读性、可维护性和避免潜在的错误



#### 6.2 自定义模块

- **创建模块**：将相关代码放入一个 `.py` 文件中。

- **导入模块**：使用 `import` 语句在其他文件中导入模块。

- **使用模块**：通过 `模块名.函数名` 或 `from 模块名 import 函数名` 的方式使用模块中的功能



---

### 7、第三方包

`pip install package`

>以下是一些建议安装的包：
>
>- **NumPy**：用于高效的数值计算，提供强大的数组操作功能，是科学计算的基础
>- **Pandas**：提供数据结构和数据分析工具，特别适用于处理和分析结构化数据
>- **Matplotlib**：用于生成各种静态、动态和交互式的图表和可视化效果，帮助你理解数据
>- **Scikit-learn**：提供简单高效的机器学习工具，包括分类、回归、聚类等算法，适合进行数据挖掘和分析
>- **SQLAlchemy**：功能强大的数据库工具包，提供数据库连接和操作的抽象层，支持多种数据库系统
>- **Django**：高级Web框架，鼓励快速开发和设计简洁实用，适合构建复杂的Web应用
>- **Flask**：轻量级Web框架，适用于构建小型应用程序，灵活且易于扩展
>- **Requests**：简化HTTP请求发送，适用于与网络资源交互，代码简洁易读
>- **Beautiful Soup**：用于从HTML和XML文件中提取数据，适合进行网页抓取和解析
>- **Scrapy**：强大的网络抓取框架，适用于大规模爬虫任务，支持异步处理
>- **TensorFlow**或**PyTorch**：如果你对深度学习和人工智能感兴趣，这两个框架是业界广泛使用的工具
>- **OpenCV**：用于计算机视觉任务，提供丰富的图像处理功能
>- **PyGame**：用于开发视频游戏的库，提供图像、声音和输入处理功能
>



---

### 8、可视化

#### 8.1 json数据格式的转换

**什么是json**

- 一种轻量级的数据交互格式，可以按照指定的格式去组织和封装数据
- 本质上是一个带有特定格式的字符串
- 负责不同编程语言中的数据传递和交互

相互转换：

```python
import json
data = [{"name":"zyh","age":19},{"name":"aaa","age":111}]
data = json.dumps(data)#py转json
data = json.loads(data)#json转py
```



如果有中文可以加上`ensure_ascii=False` 来确保中文正常转换

---

#### 8.2 pyecharts入门



先欠着后面来补



### 9、类和对象

定义：

成员变量&成员函数

```python
class Student:
    def __init__(self, name , age ):
        self.name = name
        self.age = age
    def greet(self):
        print( f"你好，我是{self.name},刚满{self.age}岁)
```

成员方法的定义：

```python
def F_Name(self,形参1，形参2......):
    方法体
```

#### 9.1 self

在 Python 中，`self` 就是对当前实例对象的引用，是你在实例方法中“跟自己打招呼”的方式。每当你调用一个实例方法时，Python 会自动把这个实例传递给方法中的 `self` 参数，从而让你可以通过 `self` 访问和修改对象的属性，或者调用其他方法。虽然 `self` 只是一个约定，并非 Python 的关键字，但它帮助你把每个对象的数据和行为组织在一起，就像每个人都有一个专属的身份证一样，让代码既清晰又易于维护



#### 9.2 魔术方法

##### `__str__方法`

当使用 `str()` 函数或 `print()` 打印对象时，会调用该方法以获得一个面向用户、易于阅读的字符串表示

```python
class Stu():

    def __init__(self,name,major):

        self.name = name

        self.major = major

    def __str__(self):

        return f"{self.name},{self.major}"
zyh = Stu("zyh","cs")

print(zyh)
```



##### `__lt__方法`



##### `__eq__方法`

比较两个对象是否相等



#### 9.3 封装

 并非所有属性和行为都开放（内部可以访问，外界无法访问）

私有成员变量
私有成员方法

| 定义方式         | 解释                                                      | 例子            |
| :--------------- | :-------------------------------------------------------- | :-------------- |
| 单下划线 `_var`  | **“这是内部使用的变量，不要随便动！”** 但其实还是可以访问 | `_my_var = 42`  |
| 双下划线 `__var` | **名字重整**（name mangling），让外部访问起来非常不方便   | `__my_var = 42` |

实际上是进行了 **名字重整** 

例如：一个 `stu`类的`__zyh`实际上被重整为 `_stu__zyh`

```python
class stu:
    def __init__(self,name,age):
        self.name = name
        self.__age = age
stu_zyh = stu("zyh",18)
print(stu_zyh.__age)#不可以
print(stu_zyh._stu__age)#可以
```

#### 9.4 继承











































