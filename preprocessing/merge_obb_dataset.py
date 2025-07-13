'''
Author: BTZN0325 sunjiahui@boton-tech.com
Date: 2024-03-07 10:53:05
LastEditors: BTZN0325 sunjiahui@boton-tech.com
LastEditTime: 2024-03-29 14:22:07
Description: 
'''
import argparse
import os
import os.path as osp


def check_dirs(dataset_dir):
    assert osp.exists(dataset_dir), f"{dataset_dir} is not existed!"
    train_img_dir = osp.join(dataset_dir, "images", "train")
    val_img_dir = osp.join(dataset_dir, "images", "val")
    train_label_dir = osp.join(dataset_dir, "labels", "train")
    val_label_dir = osp.join(dataset_dir, "labels", "val")
    train_original_label_dir = osp.join(dataset_dir, "labels", "train_original")
    val_original_label_dir = osp.join(dataset_dir, "labels", "val_original")

    assert osp.exists(train_img_dir), f"{train_img_dir} is not existed!"
    assert osp.exists(val_img_dir), f"{val_img_dir} is not existed!"
    assert osp.exists(train_label_dir), f"{train_label_dir} is not existed!"
    assert osp.exists(val_label_dir), f"{val_label_dir} is not existed!"
    assert osp.exists(train_original_label_dir), f"{train_original_label_dir} is not existed!"
    assert osp.exists(val_original_label_dir), f"{val_original_label_dir} is not existed!"

    return train_img_dir, val_img_dir, train_label_dir, val_label_dir, train_original_label_dir, val_original_label_dir


def merge_dataset(dataset_dirs, dst_dir):
    if not osp.exists(dst_dir): os.makedirs(dst_dir, exist_ok=True)
    dst_train_img_dir = osp.join(dst_dir, "images", "train")
    dst_val_img_dir = osp.join(dst_dir, "images", "val")
    dst_train_label_dir = osp.join(dst_dir, "labels", "train")
    dst_val_label_dir = osp.join(dst_dir, "labels", "val")
    dst_train_original_label_dir = osp.join(dst_dir, "labels", "train_original")
    dst_val_original_label_dir = osp.join(dst_dir, "labels", "val_original")

    if not osp.exists(dst_train_img_dir): os.makedirs(dst_train_img_dir, exist_ok=True)
    if not osp.exists(dst_val_img_dir): os.makedirs(dst_val_img_dir, exist_ok=True)
    if not osp.exists(dst_train_label_dir): os.makedirs(dst_train_label_dir, exist_ok=True)
    if not osp.exists(dst_val_label_dir): os.makedirs(dst_val_label_dir, exist_ok=True)
    if not osp.exists(dst_train_original_label_dir): os.makedirs(dst_train_original_label_dir, exist_ok=True)
    if not osp.exists(dst_val_original_label_dir): os.makedirs(dst_val_original_label_dir, exist_ok=True)

    for dataset_dir in dataset_dirs:
        train_img_dir, val_img_dir, train_label_dir, val_label_dir, train_original_label_dir, val_original_label_dir = check_dirs(dataset_dir)

        # 使用 find 和 xargs 复制train图片文件
        find_cmd = f"find {train_img_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_train_img_dir}"
        print(find_cmd)
        os.system(find_cmd)

        # 使用 find 和 xargs 复制val图片文件
        find_cmd = f"find {val_img_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_val_img_dir}"
        print(find_cmd)
        os.system(find_cmd)

        # 使用 find 和 xargs 复制train标签文件
        find_cmd = f"find {train_label_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_train_label_dir}"
        print(find_cmd)
        os.system(find_cmd)

        # 使用 find 和 xargs 复制val标签文件
        find_cmd = f"find {val_label_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_val_label_dir}"
        print(find_cmd)
        os.system(find_cmd)

        # 使用 find 和 xargs 复制train原始标签文件
        find_cmd = f"find {train_original_label_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_train_original_label_dir}"
        print(find_cmd)
        os.system(find_cmd)

        # 使用 find 和 xargs 复制val原始标签文件
        find_cmd = f"find {val_original_label_dir} -maxdepth 1 -print0 | xargs -0 cp -v -t {dst_val_original_label_dir}"
        print(find_cmd)
        os.system(find_cmd)

    print("merge done!")


def main():
    parser = argparse.ArgumentParser(description='Merge multiple datasets into one')
    parser.add_argument('--src_dirs', nargs='+', required=True, 
                        help='Source dataset directories to merge')
    parser.add_argument('--dst_dir', required=True, 
                        help='Destination directory for merged dataset')
    
    args = parser.parse_args()
    
    # 验证源目录是否存在
    for src_dir in args.src_dirs:
        if not osp.exists(src_dir):
            print(f"Error: Source directory {src_dir} does not exist!")
            return
    
    print(f"Merging datasets from: {args.src_dirs}")
    print(f"Output directory: {args.dst_dir}")
    
    merge_dataset(args.src_dirs, args.dst_dir)


if __name__ == "__main__":
    main()