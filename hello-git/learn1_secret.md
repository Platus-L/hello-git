~~chatgpt 陪伴全程QAQ~~
## Part 函数的学习
### I open():读写文件函数
#### 1. 打开文件 
```python
    open("1.txt", "r") #打开文件,返回文件
```

 | "x"     | 含义      |
|---|---------|
|"r" |  读模式|
|"w" |  写模式,覆盖or新建|
|"a" |     add|
|"r+" | 读写|
|"b" |  二进制模式,组合技("rb","rw")|

#### 2.读文件

```python
#一次性读取全部内容
f = open("1.txt", "r", econding = "utf-8")
content = f.read()
print = (content)
f.close()
```
```python
#按行读取
f = open("1.txt", "r", ecoding = "utf-8")
for line in f:
    print(line.strip()) #去掉换行符
f.close()
```
```python
#readlines()函数，返回列表
f = open("1.txt", "r", ecoding = "utf-8")
lines = f.readlines()
print(lines)
f.close()
```

### II read():读取文件
```python
f.read(size = -1) # size : 读取的字符数
```

### III write():写入函数
向文件写入 **字符串**（文本模式）或 **字节**（二进制模式）。
```python
f.write(string)
```
注意：写入 vs **追加**写入
```python
with open("1.txt", "w", ecoding = "utf-8") as f:
    f.write("Hello world\n")
# 写入 = 覆盖
```
```python
with open("1.txt", "a", ecoding = "utf-8") as f:
    f.write("Hello world\n")
#插入
```
### IV chr():字符转化函数
·功能：将**unicode码**转成**字符**
```python 
chr(i) #i：整数，表示 Unicode 码点（必须在 0 ~ 0x10FFFF 范围内）
```
返回值：对应字符（字符串类型，长度为 1）
### V int():整数转化函数
```python
int(x, base = 10) # i：字符or数字，base：进制
```
注：`ord()`:字符->数字
### VI split():str拆分成列表
```python
str.split(sep = None, maxsplit = -1)
```
&middot; `sep`： 分隔符，默认*None*，表示按任意空白字符分割
·`maxsplit`：最大分割符，默认`-1`表示全部分割
**注意：**`split('\n')`与`readlines()`的区别，后者分割后带`'\n'`
