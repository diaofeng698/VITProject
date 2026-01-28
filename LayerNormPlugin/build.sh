#!/bin/bash

# 清理旧的构建文件
make clean

# 编译
make -j

# 库文件已经在当前目录生成,无需复制