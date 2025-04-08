'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2024-11-14 01:07:56
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-11-19 01:17:32
Description: 
'''

import argparse
from pathlib import Path
import random
import shutil


def split_yolo(yolo_root_path: str, ratio: list = [0.8, 0.2, 0]):
    """
    Splits the YOLO dataset into training, validation, and test sets.

    Args:
        yolo_root_path (str): The root directory path of the YOLO dataset.
        ratio (list): The ratio of the training, validation, and test sets.
    
    Example:
        ```python
        split_yolo("path/to/YOLO", ratio=[0.8, 0.2, 0])
        ```
    
    Notes:
        The directory structure assumed for the YOLO dataset:
        
            - YOLO
                ├─ images
                │   ├─ 1.jpg
                │   ├─ ...
                │   └─ n.jpg
                ├─ labels
                │   ├─ 1.txt
                │   ├─ ...
                └─  └─ n.txt

        After execution, the function will organize the labels into:
        
            - YOLO
                ├─ images
                │   ├─ train
                │   ├─ val
                │   └─ test (if test ratio is not zero)
                └─ labels
                    ├─ train
                    ├─ val
                    └─ test (if test ratio is not zero)

    """
    # Check if the ratio is valid
    if sum(ratio) != 1:
        raise ValueError("The sum of the ratio values must be equal to 1.")

    # Convert root path to Path object
    yolo_root_path = Path(yolo_root_path)
    yolo_image_path = yolo_root_path / "images"
    yolo_label__path = yolo_root_path / "labels"
    
    # Define target directories for images and labels
    image_train_path = yolo_image_path / "train"
    image_val_path = yolo_image_path / "val"
    label_train_path = yolo_root_path / "labels" / "train"
    label_val_path = yolo_root_path / "labels" / "val"
    
    # Only create test paths if test ratio is not zero
    if ratio[2] > 0:
        image_test_path = yolo_image_path / "test"
        label_test_path = yolo_root_path / "labels" / "test"
    else:
        image_test_path = None
        label_test_path = None

    # Create directories if they do not exist
    for path in [image_train_path, image_val_path, label_train_path, label_val_path]:
        path.mkdir(parents=True, exist_ok=True)
    if image_test_path and label_test_path:
        image_test_path.mkdir(parents=True, exist_ok=True)
        label_test_path.mkdir(parents=True, exist_ok=True)

    # Get list of all image files and shuffle them for random splitting
    # List of supported image extensions
    extensions = ["*.jpg", "*.jpeg", "*.JGP", "*.JPEG", "*.png", "*.PNG", 
                  "*.bmp", "*.BMP", "*.dng", "*.mpo", "*.tif", "*.tiff", "*.webp", "*.pfm"]
    images = []
    for ext in extensions:
        images.extend(yolo_image_path.glob(ext))
    random.shuffle(images)

    # Calculate the number of images for each split
    total_images = len(images)
    train_count = int(total_images * ratio[0])
    val_count = int(total_images * ratio[1])
    test_count = total_images - train_count - val_count

    # Adjust train_count or val_count if test_count < 0
    if test_count < 0:
        val_count += test_count  # Reduce val_count to ensure total = total_images
        test_count = 0

    # Ensure no image is left unallocated
    allocated_count = train_count + val_count + test_count
    if allocated_count < total_images:
        train_count += total_images - allocated_count
    
    # Split images into train, val, and test sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:] if ratio[2] > 0 else []

    # Function to move files
    def move_files(image_list, image_dest, label_dest):
        for image_path in image_list:
            # Move image file
            shutil.move(image_path, image_dest / image_path.name)
            # Move corresponding label file
            label_path = yolo_label__path / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.move(label_path, label_dest / label_path.name)

    # Move files to their respective folders
    move_files(train_images, image_train_path, label_train_path)
    move_files(val_images, image_val_path, label_val_path)
    if ratio[2] > 0:
        move_files(test_images, image_test_path, label_test_path)

    print(f"Dataset split complete: {train_count} train, {val_count} val, {test_count} test.")


def main():
    parser = argparse.ArgumentParser(description="split YOLO dataset")
    parser.add_argument('--data', type=str, help='path of YOLO dataset')
    parser.add_argument('--ratio', nargs='+', default=[0.8, 0.2, 0], help='path of output voc dataset')
    args = parser.parse_args()
    # Convert the ratio to a list of floats
    ratio = list(map(float, args.ratio))
    split_yolo(args.data, ratio)


if __name__ == "__main__":
    main()