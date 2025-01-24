'''
Author: BTZN0325 sunjiahui@boton-tech.com
Date: 2024-01-09 09:45:38
LastEditors: BTZN0325 sunjiahui@boton-tech.com
LastEditTime: 2024-02-02 10:41:13
Description: 用于生成X光小图缺陷，主要逻辑是将小图缺陷抠图作为模板，再将其粘贴至不同的背景图上，同时会生成对应的xml标签。
'''

import os
import os.path as osp
from PIL import Image
from paste_images import *
from stitching_images import parse_voc_annotation, create_voc_xml
from time import time
from copy import deepcopy


def random_select_bg(bg_folder, filename):
    """
    用于随机选择背景图，由于背景图可能太大太多，因此不提前载入内存
    Args:
        bg_folder(str): 存放背景图的文件夹
        filename(str): 生成图像的文件名，用于替换原图的文件名，是通过timestamp生成的
    Return:
        PIL.Image
        bg_name: 原图的文件名
    """
    bg_name = random.choice(os.listdir(bg_folder))
    bg = Image.open(os.path.join(bg_folder, bg_name))
    bg.filename = filename

    return bg, bg_name


def random_select_sub(sub_images, num, label):
    """
    用于随机选择缺陷模板，模板一般较小，数量也不多，因此提前载入内存。
    Args:
        sub_images(List[PIL.Image, ...]): 存放子图的文件夹，一般来说
        num(int): 随机挑选子图的数量
        label_list(str): 标签，用于给目标赋标签
    Return:
        List[Target]
    """
    sub_images_sample = random.sample(sub_images, num)
    sub_images_select = []
    for sub_img in sub_images_sample:
        # sub_img = rotate_90(sub_img)
        sub_img = rotate(sub_img)
        # sub_img = cj(sub_img)
        sub_img = flip(sub_img)
        sub_images_select.append(Target(sub_img, label))

    return sub_images_select


def generate_random_integers(total_sum, length):
    """
    用于生成length个随机整数，其和为total_num。
    例如total_num = 10，tength = 3，那么可以返回的list有 [1, 1, 8], [1, 2, 7], ...
    """
    if total_sum < length:
        raise ValueError("total_sum must be greater than or equal to length")

    # 生成 len-1 个介于 0 和 total_sum 之间的随机整数
    random_points = sorted(random.sample(range(1, total_sum), length - 1))
    
    # 在列表的开始和结尾添加 0 和 total_sum
    random_points = [0] + random_points + [total_sum]
    
    # 计算相邻分界点之间的差值来得到最终的数字列表
    return [random_points[i+1] - random_points[i] for i in range(length)]


def load_images(images_dict):
    """
    用于图片预加载。
    Args:
        images_dict(dict): 格式为 {'标签名': '存放图片的路径'}
    Return:
        images_total(dict): 格式为 {'标签名': [PIL.Image, ...]}
    """
    images_total = {}
    for label in images_dict:
        path = images_dict[label]
        if not osp.exists(path): raise f"{path} is not existed! Please check your image_dict!"
        images_total[label] = []
        for img_name in os.listdir(path):
            images_total[label].append(Image.open(osp.join(path, img_name)))
    return images_total


def gen_images(bg_folder, bg_label_folder, num, max_target_num, sub_images_total, out_dir):
    for gen_num in range(num):
        target_num_total = random.randint(len(sub_images_dict), max_target_num)                      # 一张背景图上随机粘贴的缺陷个数
        target_num_list = generate_random_integers(target_num_total, len(sub_images_dict))           # 每种缺陷要粘贴的个数
        timestamp = int(time()*1000)
        filename = f"{timestamp:13d}_{gen_num}.jpg"

        bg, bg_name = random_select_bg(bg_folder, filename)
        
        # 粘贴的区域，x_range代表宽方向，y_range代表高方向
        x_range = (0, bg.size[0])
        y_range = (100, bg.size[1]-2*100)
        
        # 读取背景图对应的标签，格式为 [{'class': cls_name, 'bbox': (xmin, ymin, xmax, ymax)}, ...]
        if osp.exists(osp.join(bg_label_folder, bg_name.replace('jpg', 'xml'))):
            objects = parse_voc_annotation(osp.join(bg_label_folder, bg_name.replace('jpg', 'xml')))
        else:
            objects = []
        sub_images_select = []
        for target_num, sub_images_label in zip(target_num_list, sub_images_total):
            sub_images_select.extend(random_select_sub(deepcopy(sub_images_total[sub_images_label]), target_num, sub_images_label))
        bg = paste_image(bg, sub_images_select, x_range, y_range, objects)
        if bg is None:
            continue
        for sub_img in sub_images_select:
            object = {
                "class": sub_img.label,
                "bbox": tuple(sub_img.bbox)
            }
            objects.append(object)
        create_voc_xml(objects, out_dir, bg.filename, bg.size[0], bg.size[1])
        bg.save(osp.join(out_dir, bg.filename))

        # bg_enhance = sub_cj(deepcopy(bg))
        # bg_enhance.save(osp.join(out_dir, bg.filename))

# 小图的数据增强
rotate = Rotation(degree=180, mode='expand')
# rotate_90 = Rotation(prob=1.0, degree=90, mode='expand')   # 因为fall缺陷小图的方向和原图不一样，所以旋转90度保持和图像方向一致
# cj = ColorJitter(prob=0.5, min_factor=0.8, max_factor=1.2)
flip = Flip()
# 生成图的数据增强
# sub_cj = ColorJitter(min_factor=0.8, max_factor=1.2)


if __name__ == "__main__":
    bg_root = '/data/bt/xray_fanglun/LabeledData/20240202'
    bg_folder = osp.join(bg_root, "images")                    # 背景图
    bg_label_folder = osp.join(bg_root, "voc_labels_old")          # 背景图标签
    sub_images_dict = {
                       'embedding_cikou': '/data/bt/xray_fanglun/material/20240110/embedding_cikou',       # 扣下来的小图，用于粘贴至背景图，无标签
                    #    'fall':      '/data/bt/xray_fanglun/material/20240110/fall'
                       }             
    out_dir = '/data/bt/xray_fanglun/LabeledData/20240202_2'                                    # 输出目录，会将粘贴图以及对应的标签保存至此
    num = 100                                                                                   # 生成图像的数量     
    max_target_num = 5                                                                        # 一张图上粘贴缺陷的最大数量
    sub_images_total = load_images(sub_images_dict)

    gen_images(bg_folder, bg_label_folder, num, max_target_num, sub_images_total, out_dir)
