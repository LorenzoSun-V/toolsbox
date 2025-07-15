# description: This script gathers samples from multiple directories, merging them into a single directory.

import os
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
                # Construct the destination file path
                dest_file = os.path.join(dest_dir, os.path.relpath(file, src_dir))
                dest_file_dir = os.path.dirname(dest_file)
                
                # Create the destination directory if it does not exist
                os.makedirs(dest_file_dir, exist_ok=True)
                
                # Copy the file to the destination directory
                shutil.copy2(file, dest_file)
                print(f"Copied {file} to {dest_file}")


if __name__ == "__main__":
    source_dirs = [
        "/data/nofar/material/liandongUgu/2025-07-09/person_behavior_labeling/0711",
        "/data/nofar/material/liandongUgu/2025-07-09/person_behavior_labeling/0714",
    ]
    destination_dir = "/data/nofar/material/liandongUgu/2025-07-09/person_behavior_labeling/merged"
    gather_samples(source_dirs, destination_dir)