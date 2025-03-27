# 一、基础内容

## 1 注释

单行 `//`

多行 `/*    */`

>与java相同



## 2 常量

用于记录不可更改的数据

- `#define 宏常量` 通常在文件最上方
- `const 变量`  如 `const int week=7` 



## 3 标识符命名

1. 不能为关键字
2. 数字，字母，下划线
3. 开头为字母或下划线



## 4 数据类型

1. 整型
	>`int` 4
	>`long` 4
	>
	>`longlong` 8
	>
	>`short` 2

2. 浮点型
	表示小数

	>`float` 单精度 4字节 7位有效数字
	>`double` 双精度 8字节

3. 字符型
	`char` 如`char ch = 'a';`

	>使用单引号
	>单引号内只能有一个字符
	>并非直接存储字符而是转为ASCII码
	>a–97   A–65
	>b–98   B–66

4. 转义字符
	`\n` 换行
	`\t` 水平制表

5. 字符串
	`string str='blablabla'` 
	需要包含头文件 `#include<string>`

6. 布尔类型
	`true`
	`false`

## 5 运算符

### 5.1 算数运算符
假设变量 A 的值为 10，变量 B 的值为 20，则：

| 运算符 | 描述                                                         | 实例             |
| :----- | :----------------------------------------------------------- | :--------------- |
| +      | 把两个操作数相加                                             | A + B 将得到 30  |
| -      | 从第一个操作数中减去第二个操作数                             | A - B 将得到 -10 |
| *      | 把两个操作数相乘                                             | A * B 将得到 200 |
| /      | 分子除以分母（整数相除取整）                                 | B / A 将得到 2   |
| %      | 取模运算符，整除后的余数                                     | B % A 将得到 0   |
| ++     | [自增运算符](https://www.runoob.com/cplusplus/cpp-increment-decrement-operators.html)，整数值增加 1 | A++ 将得到 11    |
| --     | [自减运算符](https://www.runoob.com/cplusplus/cpp-increment-decrement-operators.html)，整数值减少 1 | A-- 将得到 9     |

> 区分：
>
> `b=++a` 表示a先加1，再赋值给b
> `b=a++` 表示先赋值给b，a再加1

### 5.2 比较运算符

`==`

`!=`

`>` `>=`
`<` `<=`



### 5.3 逻辑运算符

`!` 非
`&&` 与（且）

`||` 或



### 5.4 杂项运算符

- `sizeof` 返回变量的大小
- `&` 返回变量的地址
- `condition ？X:Y` condition为真则值为X，反之Y



## 6 选择与循环

### 6.1 if

```cpp
if(codition_A)
{
    blablabla;
}
else
{
    blablabla;
}
```

每一个else与最近的if匹配



### 6.2 switch

```cpp
switch(表达式)
{
    case 结果1:语句1;break;
    case 结果2:语句2;break;
        ......
}

```



### 6.3 while

```cpp
while(条件)
{
    blablabla
}
```

注意死循环



### 6.4 do-while

```cpp
do
{
    blablabla
}
while(条件)
```

与while不同在先执行一次再进行判断



### 6.5 for

```cpp
for(初始表达式;条件表达式;末尾循环体)
{
    循环体；
}
//ex:
for(int i=1;i<=100;i++)
{
    sum+=1;
}
```

### 6.6 跳转

#### break

退出最近的循环，不影响外层

#### continue

跳过本次循环，执行下一次循环

#### `goto`

无条件跳转
`goto FLAG;FLAG:blablabla`

---

## 7 数组

*所有的数组都是由连续的内存位置组成。最低的地址对应第一个元素，最高的地址对应最后一个元素*

### 7.1 声明与初始化

```cpp
//type arrayName [arratSize]={a1,a2,a3... }
int score [6]={1,1,4,5,1,4}
```

未赋值用0填补

对于字符型数组，定义长度要比实际长度多1 用于存放字符串结束标志  `\0` 

数组的索引：下标，从0开始



### 7.2 二维数组

定义方式：`type arrayName[行数][列数]={{}{}{}...} `



### 数组应用实例——冒泡排序

```cpp
//比较相邻元素，若第一个大则交换它们
//重复上述操作，找到最大值
//重复上述操作，直到不需要比较
void sort(int arr[])
{
    int len = sizeof arr[];
    for(int i=0;i<)
}
```

