'''
Author: BTZN0325 sunjiahui@boton-tech.com
Date: 2024-01-08 09:08:19
LastEditors: BTZN0325 sunjiahui@boton-tech.com
LastEditTime: 2024-01-23 11:42:13
Description: 将一张图像粘贴至另一张图像。
'''
from utils import *


def paste_image(
        bg, 
        sub_images, 
        x_range = None, 
        y_range = None, 
        bg_targets = [],
        fill_mode = None,
        alpha = 1.0):
    """
    Describe:
        将sub_images里的子图，按照坐标粘贴至bg上。
    Args:
        bg(PIL.Image): 背景图。
        sub_images(List[Target]): 子图。
        x_range(tuple): 子图粘贴至背景图的横坐标。默认是None，如果是None, 那么会把背景图的宽作为取值范围。
        y_range(tuple): 子图粘贴至背景图的纵坐标。默认是None，如果是None, 那么会把背景图的高作为取值范围。
        fill_mode(str): 填充子图的方式，只能为 max, min, mean。
        alpha(float): 子图的不透明度。默认是1.0。
    Return:
        bg(PIL.Image)
    """

    if x_range is None:
        x_range = (0, bg.size[0])
    if y_range is None:
        y_range = (0, bg.size[1])
    
    sub_images_size = [sub.img.size for sub in sub_images]
    coords, bboxes = random_coords(x_range, y_range, sub_images_size, bg_targets)
    if len(coords) != len(sub_images):
        return None
    
    # 将小图粘贴到大图上
    for target, coord, bbox in zip(sub_images, coords, bboxes):
        
        if fill_mode is not None:
            # 获取背景区域
            bg_section = bg.crop((coord[0], coord[1], coord[0] + target.img.size[0], coord[1] + target.img.size[1]))
            # 调整小图的颜色
            target.img = adjust_colors_to_background(target.img, bg_section, fill_mode)
        
        # 调整小图透明度
        target.img = target.img.convert("RGBA")
        datas = target.img.getdata()
        newData = []
        if alpha != 1:
            for item in datas:
                # 改变所有像素的透明度
                newData.append((item[0], item[1], item[2], int(item[3] * alpha)))
            target.img.putdata(newData)

        bg.paste(target.img, coord, target.img)
        # 由于list是可变类型，所以传的是引用
        target.bbox = bbox

    return bg
