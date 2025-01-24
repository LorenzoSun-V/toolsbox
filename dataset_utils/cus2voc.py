'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-21 03:03:45
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2023-11-21 03:16:00
Description: 
'''
import argparse
import json
import os
import os.path as osp
import time
from glob import glob
from PIL import Image
from tqdm import tqdm
from datetime import date

import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET


def custom_to_voc2017(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_path = data['imagePath']
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = osp.dirname(output_dir)
    ET.SubElement(root, 'filename').text = osp.basename(image_path)
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(image_width)
    ET.SubElement(size, 'height').text = str(image_height)
    ET.SubElement(size, 'depth').text = '3'

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        xmin = str(points[0][0])
        ymin = str(points[0][1])
        xmax = str(points[1][0])
        ymax = str(points[1][1])

        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = label
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0'
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = xmin
        ET.SubElement(bndbox, 'ymin').text = ymin
        ET.SubElement(bndbox, 'xmax').text = xmax
        ET.SubElement(bndbox, 'ymax').text = ymax

    xml_string = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_string)
    formatted_xml = dom.toprettyxml(indent='  ')

    with open(output_dir, 'w') as f:
        f.write(formatted_xml)


def main():
    parser = argparse.ArgumentParser(description="custom data 2 voc")
    parser.add_argument('--data', type=str, help='path of custom dataset')
    parser.add_argument('--output', type=str, help='path of output voc dataset')
    args = parser.parse_args()

    json_dirs = sorted(glob(osp.join(args.data, "*.json")))
    for json_dir in tqdm(json_dirs):
        file_name_stem = osp.basename(json_dir)[:-5]
        output_dir = osp.join(args.output, f"{file_name_stem}.xml")
        custom_to_voc2017(json_dir, output_dir)


if __name__ == "__main__":
    main()