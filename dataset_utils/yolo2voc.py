#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO to VOC XML converter

Refactored by ChatGPT:
- Supports multiple image extensions
- Uses pathlib for cleaner path handling
- Modular functions: argument parsing, class loading, image lookup, XML building
- Adds simple logging for diagnostics
"""
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert YOLO label files to VOC-style XML annotations"
    )
    parser.add_argument(
        '--voc-label-list',
        type=Path,
        help='Path to VOC label list (one class name per line).'
    )
    parser.add_argument(
        '--root-dir',
        type=Path,
        help='Root directory for YOLO dataset (contains labels/ and images/).'
    )
    parser.add_argument(
        '--yolo-dir',
        type=Path,
        default=None,
        help='Optional path to label .txt files (overrides root/labels/).'
    )
    parser.add_argument(
        '--img-dir',
        type=Path,
        default=None,
        help='Optional path to images (overrides root/images/).'
    )
    return parser.parse_args()


def load_classes(label_list: Path) -> list:
    return label_list.read_text(encoding='utf-8').splitlines()


def find_image(img_dir: Path, img_id: str, exts=None) -> Path:
    exts = exts or ['jpg', 'jpeg', 'png', 'bmp', 'tif']
    for ext in exts:
        candidate = img_dir / f"{img_id}.{ext}"
        if candidate.exists():
            return candidate
    return None


def unconvert(x, y, w, h, width, height):
    """
    Convert YOLO normalized bbox (x_center, y_center, w, h) to VOC (xmin, ymin, xmax, ymax)
    """
    xmin = int((x - w/2) * width)
    ymin = int((y - h/2) * height)
    xmax = int((x + w/2) * width)
    ymax = int((y + h/2) * height)
    return xmin, ymin, xmax, ymax


def build_voc_xml(
    img_id: str,
    img_filename: str,
    size: tuple,
    annotations: np.ndarray,
    classes: list
) -> bytes:
    width, height, depth = size
    root = Element('annotation')

    # folder and filename
    SubElement(root, 'folder').text = img_filename
    SubElement(root, 'filename').text = img_filename

    # source
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # size
    size_el = SubElement(root, 'size')
    SubElement(size_el, 'width').text = str(width)
    SubElement(size_el, 'height').text = str(height)
    SubElement(size_el, 'depth').text = str(depth)

    SubElement(root, 'segmented').text = '0'

    # objects
    for ann in annotations:
        cls_id, x, y, w, h = ann
        xmin, ymin, xmax, ymax = unconvert(x, y, w, h, width, height)
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = classes[int(cls_id)]
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    return tostring(root, pretty_print=True)


def xml_transform(root: Path, classes: list, yolo_dir: Path, img_dir: Path):
    label_dir = yolo_dir or (root / 'labels')
    image_dir = img_dir or (root / 'images')
    out_dir = root / 'Annotations'
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_file in tqdm(label_dir.glob('*.txt')):
        img_id = label_file.stem
        xml_path = out_dir / f"{img_id}.xml"
        if xml_path.exists():
            logging.info(f"Skipping existing: {xml_path}")
            continue

        img_file = find_image(image_dir, img_id)
        if not img_file:
            logging.warning(f"Image not found for ID {img_id}")
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            logging.error(f"Failed to read image: {img_file}")
            continue

        h, w, c = img.shape
        try:
            data = np.loadtxt(label_file).reshape(-1, 5)
        except Exception:
            logging.warning(f"No annotations in {label_file}")
            data = np.zeros((0, 5))

        xml_bytes = build_voc_xml(
            img_id,
            img_file.name,
            (w, h, c),
            data,
            classes
        )
        xml_path.write_bytes(xml_bytes)
        logging.info(f"Wrote: {xml_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    classes = load_classes(args.voc_label_list)
    xml_transform(
        root=args.root_dir,
        classes=classes,
        yolo_dir=args.yolo_dir,
        img_dir=args.img_dir
    )


if __name__ == '__main__':
    main()
