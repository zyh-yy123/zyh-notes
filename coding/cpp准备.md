## 一、单文件编译运行

假设项目目录中仅有 `main.cpp`。

```
 1. 切换到源文件目录
cd path/to/your/project

编译（指定 C++17 标准 + 警告 + 调试信息）
g++ -std=c++17 -Wall -g main.cpp -o main.exe

# 3. 运行可执行文件
./main.exe       # Linux/macOS/WSL
main.exe         # Windows PowerShell/CMD
```

- `-std=c++17`：使用 C++17 标准 [Visual Studio Code](https://code.visualstudio.com/docs/cpp/config-linux?utm_source=chatgpt.com)
- `-Wall`：开启所有常见警告 [GeeksforGeeks](https://www.geeksforgeeks.org/vs-code-compile-and-run-in-c/?utm_source=chatgpt.com)
- `-g`：生成调试信息，供 GDB 使用 [Visual Studio Code](https://code.visualstudio.com/docs/cpp/config-linux?utm_source=chatgpt.com)
- `-o main.exe`：指定输出文件名 [Stack Overflow](https://stackoverflow.com/questions/73659688/how-to-run-c-program-from-terminal-vs-code?utm_source=chatgpt.com)
- 或者直接 `g++ -o name.exe`然后 `./name.exe`运行

------

## 二、多文件项目



目录结构示例：

```
css复制编辑project/
├─ src/
│  ├─ main.cpp
│  ├─ utils.cpp
│  └─ utils.h
```

### 方法 1：一次性编译链接

```
cd project/src
g++ -std=c++17 -Wall -g *.cpp -o app.exe
./app.exe
```

- `*.cpp` 会匹配所有源文件，并在一条命令中完成编译与链接 [GeeksforGeeks](https://www.geeksforgeeks.org/vs-code-compile-and-run-in-c/?utm_source=chatgpt.com)。

### 方法 2：分步骤编译与链接

1. **编译生成目标文件**

	```
	bash复制编辑g++ -std=c++17 -Wall -g -c main.cpp    # 生成 main.o
	g++ -std=c++17 -Wall -g -c utils.cpp   # 生成 utils.o
	```

	`-c`：仅编译，不链接，生成 `.o`（Linux/macOS）或 `.obj`（Windows）文件 [GeeksforGeeks](https://www.geeksforgeeks.org/vs-code-compile-and-run-in-c/?utm_source=chatgpt.com)。

2. **链接目标文件**

	```
	bash复制编辑g++ main.o utils.o -o app.exe
	./app.exe
	```

	将所有目标文件组合生成可执行文件 [GeeksforGeeks](https://www.geeksforgeeks.org/vs-code-compile-and-run-in-c/?utm_source=chatgpt.com)。

------

## 三、VS Code 终端操作要点

- **打开终端**：`Ctrl+\``（Windows/Linux）或 `⇧⌘C`（macOS） [Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp?utm_source=chatgpt.com)。
- **当前目录**：终端默认启动于工作区根目录，可用 `cd` 切换到源代码所在文件夹 [Microsoft Learn](https://learn.microsoft.com/en-us/cpp/build/walkthrough-compiling-a-native-cpp-program-on-the-command-line?view=msvc-170&utm_source=chatgpt.com)。
- **自动化选项**：后续可配置 `tasks.json` 和 `launch.json`，实现一键编译（Ctrl+Shift+B）与调试（F5） [Ask Ubuntu](https://askubuntu.com/questions/1032647/how-can-i-compile-c-files-through-visual-studio-code?utm_source=chatgpt.com)。