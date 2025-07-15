# description: This script gathers samples from multiple directories, merging them into a single directory.

import argparse
import os
import os.path as osp
import shutil
from glob import glob
from tqdm import tqdm


def gather_samples(src_dirs, dest_dir):
    """
    Gather samples from multiple source directories into a single destination directory.
    
    Args:
        src_dirs: List of source directories.
        dest_dir: Destination directory where files will be copied.
    """

    os.makedirs(dest_dir, exist_ok=True)
    
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            print(f"Source directory {src_dir} does not exist, skipping.")
            continue
        
        # Get all files in the source directory and its subdirectories
        files = glob(os.path.join(src_dir, '**', '*.*'), recursive=True)
        for file in tqdm(files, desc=f"Gathering from {src_dir}"):
            if os.path.isfile(file):
                shutil.copy(file, dest_dir)


def main():
    parser = argparse.ArgumentParser(description='Merge multiple datasets into one')
    parser.add_argument('--src_dirs', '-s', nargs='+', required=True, 
                        help='Source dataset directories to merge')
    parser.add_argument('--dst_dir', '-d', required=True, 
                        help='Destination directory for merged dataset')
    
    args = parser.parse_args()
    
    # 验证源目录是否存在
    for src_dir in args.src_dirs:
        if not osp.exists(src_dir):
            print(f"Error: Source directory {src_dir} does not exist!")
            return
    
    print(f"Copy data from: {args.src_dirs}")
    print(f"Output directory: {args.dst_dir}")
    
    gather_samples(args.src_dirs, args.dst_dir)


if __name__ == "__main__":
    main()