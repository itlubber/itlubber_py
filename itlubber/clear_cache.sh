#!/bin/bash

PYSTRING="$(find . | grep -E "(__pycache__|\.pyc|\.pyo$)")"
IPYNBSTRING="$(find . | grep -E "(ipynb_checkpoints|\.ipynb$)")"
SO="$(find . | grep -E "(\.c$)")"

# 删除 __pycache__ 缓存文件
if [ -n "$PYSTRING" ]; then
  echo "删除以下缓存文件 :"
  echo "-----------------------------------------------------"
  echo "$PYSTRING"
  echo "-----------------------------------------------------"
  find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
else
  echo "不存在 __pycache__ 缓存文件"
fi

# 删除 ipynb_checkpoints 缓存文件
# if [ -n "$SO" ]; then
#   echo "删除以下缓存文件 :"
#   echo "-----------------------------------------------------"
#   echo "$SO"
#   echo "-----------------------------------------------------"
#   find . | grep -E "(\.c$)" | xargs rm -rf
# else
#   echo "不存在 .c 缓存文件"
# fi
