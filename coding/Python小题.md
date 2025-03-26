# 1、入门题

## 1![image-20250125145928474](../AppData/Roaming/Typora/typora-user-images/image-20250125145928474.png)



```python
#思路：把字母转换成数字，再根据奇偶性判断
#补充方法： ord() 获得字符的 Unicode 
#          a 97     A 65
 x = ord(coordinates[0]) - ord('a')
 y = ord(coordinates[1]) - ord('1')
 return (x + y) % 2 == 1
#都转成了0~7，（0,0）为黑
```

## 2

