#!/bin/bash

# 定义变量
SRC_DIRS=("/path/to/dataset1" "/path/to/dataset2" "/path/to/dataset3")
DST_DIR="/data"

# 运行Python脚本
python preprocessing/merge_obb_dataset.py --src_dirs "${SRC_DIRS[@]}" --dst_dir "$DST_DIR"