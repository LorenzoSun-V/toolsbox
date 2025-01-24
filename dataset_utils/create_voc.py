'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-08 07:43:11
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-12-16 03:14:34
Description: create train.txt and val.txt based on labeled xmls
'''
import argparse
import os
import os.path as osp
from glob import glob
import random


def get_image_extension(img_dir, stem):
    """根据图像的stem（无后缀文件名）查找对应的图像后缀"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', ".PNG", ".BMP", ".TIF"]:  # 常见的图像后缀
        img_path = osp.join(img_dir, f"{stem}{ext}")
        if osp.exists(img_path):
            return ext
    return None  # 如果没有找到匹配的图像后缀，返回None


def create_voc(args):
    xml_dirs = glob(osp.join(args.voc_anno_dir, "*.xml"))
    xml_names_stem = [f"{osp.basename(i)[:-4]}\n" for i in xml_dirs]
    random.seed(args.random_seed)
    random.shuffle(xml_names_stem)

    # xml_names = [osp.join(args.img_dir, f"{i.strip()}.jpg\n") for i in xml_names_stem]
    xml_names = []
    for stem in xml_names_stem:
        stem = stem.strip()  # 去掉换行符
        ext = get_image_extension(args.img_dir, stem)
        if ext:
            xml_names.append(f"{osp.join(args.img_dir, f'{stem}{ext}')}\n")
        else:
            print(f"Warning: No image found for {stem}")
            
    split_index = int(len(xml_names_stem) * args.train_proportion)
    if not osp.exists(args.voc_anno_list):
        os.makedirs(args.voc_anno_list, exist_ok=True)

    trainstem_txt_path = osp.join(args.voc_anno_list, "train_stem.txt")
    valstem_txt_path = osp.join(args.voc_anno_list, "val_stem.txt")
    train_txt_path = osp.join(args.voc_anno_list, "train.txt")
    val_txt_path = osp.join(args.voc_anno_list, "val.txt")

    with open(trainstem_txt_path, 'w') as f1:
        f1.writelines(xml_names_stem[:split_index])
    with open(valstem_txt_path, 'w') as f2:
        f2.writelines(xml_names_stem[split_index:])
    with open(train_txt_path, 'w') as f1:
        f1.writelines(xml_names[:split_index])
    with open(val_txt_path, 'w') as f2:
        f2.writelines(xml_names[split_index:])
    
    return 
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--voc_anno_dir', help='xml directory')
    parser.add_argument('--img_dir', help='img directory')
    parser.add_argument(
        '--voc_anno_list',
        help='In Voc format dataset, path to annotation files ids list.',
        type=str,
        default=None)
    parser.add_argument(
        '--train_proportion',
        help='the proportion of train dataset',
        type=float,
        default=0.8)
    parser.add_argument(
        '--random_seed',
        help='random_seed',
        type=int,
        default=42)
    args = parser.parse_args()
    create_voc(args)


if __name__ == '__main__':
    main()
