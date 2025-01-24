'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-10 02:17:32
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-01-23 04:01:43
Description: 
'''
import xml.etree.ElementTree as ET
import os
from os.path import join
import argparse
from glob import glob
from tqdm import tqdm


def convert_annotation(image_id, classes):
    with open('%s.xml'%(image_id), encoding="utf8", errors='ignore') as f:
        tree = ET.parse(f)
    # in_file = open('%s.xml'%(image_id))
    # tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
   
    if w==0 and h==0:
        return
    
    out_file = open('%s.txt'%(image_id), 'w')    
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except AttributeError:
            try:
                difficult = obj.find('Difficult').text
            except AttributeError:
                difficult = '0'
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        
        cls_id = classes.index(cls)
        # print("cls_id: ",cls_id)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        out_file.write(str(cls_id) + " " + " ".join([str(int(a)) for a in b]) + '\n')
    out_file.close()
        
    # text_size = os.path.getsize(str(image_id+".txt"))
    # # print(text_size)
    # if text_size == 0:
    #     os.remove(str(image_id+".txt"))


def main():
    parser = argparse.ArgumentParser(description="create classes")
    parser.add_argument('classes', nargs='+', help='which classes to do xml2yolo')
    parser.add_argument('--val-list', type=str, help='val_stem set list')
    parser.add_argument('--xml-dir', type=str, help='which classes to do xml2yolo')
    args = parser.parse_args()
    classes = args.classes

    root_path = args.xml_dir
    with open(args.val_list, 'r') as f:
        val_list = f.readlines()
        val_list = [x.strip() for x in val_list]
        xml_dirs = [join(root_path, x+".xml") for x in val_list]

    for xml_dir in tqdm(xml_dirs):
        convert_annotation(xml_dir[:-4], classes)

    root_path = os.path.dirname(args.xml_dir.rstrip(os.path.sep)) if args.xml_dir.endswith('/') else os.path.dirname(args.xml_dir)
    print(root_path)
    save_dir = join(root_path, "gts")
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    print(f"mv {args.xml_dir}/*.txt {save_dir}")
    os.system(f"mv {args.xml_dir}/*.txt {save_dir}")


if __name__ == '__main__':
    main()
