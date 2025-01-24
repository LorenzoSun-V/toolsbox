'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2024-11-14 00:54:22
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-11-18 02:48:45
Description: 
'''

import argparse
from pathlib import Path
import cv2
from ultralytics.utils import DATASETS_DIR, LOGGER, NUM_THREADS, TQDM


def convert_dota_to_yolo_obb(dota_root_path: str, class_mapping=None):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    if class_mapping is None:
        # Class names to indices mapping
        class_mapping = {
            "plane": 0,
            "ship": 1,
            "storage-tank": 2,
            "baseball-diamond": 3,
            "tennis-court": 4,
            "basketball-court": 5,
            "ground-track-field": 6,
            "harbor": 7,
            "bridge": 8,
            "large-vehicle": 9,
            "small-vehicle": 10,
            "helicopter": 11,
            "roundabout": 12,
            "soccer-ball-field": 13,
            "swimming-pool": 14,
            "container-crane": 15,
            "airport": 16,
            "helipad": 17,
        }
    elif not isinstance(class_mapping, dict):
        raise TypeError("class_mapping must be a dictionary like: \{'plane': 0, 'ship': 1 \}.")

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                
                # Clip coordinates to image boundaries
                clipped_coords = [
                    max(0, min(coords[i], image_width if i % 2 == 0 else image_height)) for i in range(8)
                ]
                # Normalize coordinates to [0, 1]
                normalized_coords = [
                    clipped_coords[i] / image_width if i % 2 == 0 else clipped_coords[i] / image_height for i in range(8)
                ]
                #! Unnormalized coordinates to [0, 1]
                # normalized_coords = [
                #     coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                # ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val", "test"]:

        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        if not image_dir.exists():
            LOGGER.warning(f"Skipping {phase} phase. Directory not found: {image_dir}")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".jpg" and image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def main():
    parser = argparse.ArgumentParser(description="DOTA to YOLO OBB converter")
    parser.add_argument('classes', nargs='+', help='which classes to do DOTA2YOLO')
    parser.add_argument('--data', type=str, help='path of DOTA dataset')
    args = parser.parse_args()
    class_mapping = {class_name: i for i, class_name in enumerate(args.classes)}
    convert_dota_to_yolo_obb(args.data, class_mapping)


if __name__ == "__main__":
    main()