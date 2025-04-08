'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-09 07:14:01
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-05-14 07:56:47
Description: 
'''
import xml.etree.ElementTree as ET
import os
from os.path import join
import argparse
from glob import glob
from tqdm import tqdm


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw if w*dw < 1 else 1
    y = y*dh
    h = h*dh if h*dh < 1 else 1
    return (x,y,w,h)

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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
   
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
        
    text_size = os.path.getsize(str(image_id+".txt"))
    # print(text_size)
    if text_size == 0:
        os.remove(str(image_id+".txt"))


def main():
    parser = argparse.ArgumentParser(description="create classes")
    parser.add_argument('--voc_label_list', help='In Voc format dataset, path to label list. The content of each line is a category.', type=str, default=None)
    parser.add_argument('--xml-dir', type=str, help='which classes to do xml2yolo')
    args = parser.parse_args()
    with open(args.voc_label_list, 'r') as f:
        classes = f.read().split()

    root_path = args.xml_dir
    xml_dirs = glob(join(root_path, "*.xml"))
    for xml_dir in tqdm(xml_dirs):
        convert_annotation(xml_dir[:-4], classes)

    root_path = os.path.dirname(args.xml_dir)
    save_dir = join(root_path, "labels")
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)

    os.system(f"mv -v {args.xml_dir}/*.txt {save_dir}")


if __name__ == '__main__':
    main()
