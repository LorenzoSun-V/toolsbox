'''
Author: BTZN0325 sunjiahui@boton-tech.com
Date: 2024-01-08 11:05:24
LastEditors: BTZN0325 sunjiahui@boton-tech.com
LastEditTime: 2024-01-23 11:39:41
Description: 
'''
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np


def iou(bbox1, bbox2):
    """
    计算两个边界框之间的交并比（Intersection over Union）。

    参数:
    bbox1, bbox2 -- 两个边界框，格式为 (x_min, y_min, x_max, y_max)

    返回:
    交并比值。
    """
    # 计算交集区域的坐标
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    # 计算交集区域的面积
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # 计算每个边界框的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集区域的面积
    union_area = bbox1_area + bbox2_area - inter_area

    # 计算交并比
    iou = inter_area / union_area if union_area != 0 else 0

    return iou, inter_area, union_area


def random_coords(x_range, 
                  y_range, 
                  sub_images_size,
                  bg_targets):
    # 生成随机坐标
    coords = []
    bboxes = []
    # 将原本背景图上的坐标加入，用于在生成缺陷坐标时，不与已有缺陷重叠
    for bg_target in bg_targets:
        coords.append(bg_target['bbox'][:2])
        bboxes.append(bg_target['bbox'])

    for index in range(len(sub_images_size)):
        cur_image_size = sub_images_size[index]

        for index in range(1000):
            x = random.randint(x_range[0], x_range[1] - cur_image_size[0])
            y = random.randint(y_range[0], y_range[1] - cur_image_size[1])
            # 新坐标的边界框
            new_bbox = (x, y, x + cur_image_size[0], y + cur_image_size[1])
            
            # 确保新坐标与已有坐标的目标不重叠
            if all(iou(new_bbox, (cx, cy, cx + sub_image_size[0], cy + sub_image_size[1]))[0] == 0 
                   for (cx, cy), sub_image_size in zip(coords, sub_images_size)):
                coords.append((x, y))
                bboxes.append(new_bbox)
                break

            if index==999: 
                print("Fail to generate coords, sub images may be too large!")

    return coords[len(bg_targets):], bboxes[len(bg_targets):]


def adjust_colors_to_background(sub_image, bg_section, fill_mode="mean"):
    """
    Describe:
        根据背景区域调整小图的颜色。
    Args:
        sub_image(PIL.Image): 子图。
        bg_section(PIL.Image): 背景图的一部分。
        fill_mode(str): 填充子图的方式，只能为 max, min, mean。
    Return:
        sub_image(PIL.Image): 调整颜色后的小图。
    """
    assert fill_mode in ["min", "max", "mean"], f"fill_mode must be 'min', 'max', 'mean'!"

    if fill_mode == "mean":
        value = np.mean(np.array(bg_section))
    elif fill_mode == "min":
        value = np.min(np.array(bg_section))
    elif fill_mode == "max":
        value = np.max(np.array(bg_section))

    sub_image_array = np.array(sub_image) * (value / 255.0)
    sub_image_array = np.clip(sub_image_array, 0, 255)

    return Image.fromarray(sub_image_array.astype('uint8'))


class Rotation(object):
    def __init__(self, prob=0.5, degree=10, mode='crop'):
        self.prob = prob
        self.degree = degree
        self.mode = mode

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            degree = random.randint(-self.degree, self.degree)
            # In PIL, the rotate method can be used directly
            if self.mode == 'crop':
                # The default behavior of PIL's rotate is to keep the image size (cropping parts of the image)
                result_img = img.rotate(degree)
            elif self.mode == 'expand':
                # To keep the full image (without cropping), expand parameter is set to True
                result_img = img.rotate(degree, expand=True)
            else:
                raise ValueError(f"mode must be in ['crop', 'expand']")
            return result_img
        else:
            return img
        

class ColorJitter(object):
    def __init__(self, prob=0.5, min_factor=0.5, max_factor=1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img

        if random.uniform(0, 1) < self.prob:
            img = self.brightness(img)
        if random.uniform(0, 1) < self.prob:
            img = self.saturation(img)
        if random.uniform(0, 1) < self.prob:
            img = self.contrast(img)
        return img

    def brightness(self, img):
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(self.min_factor, self.max_factor)
        return enhancer.enhance(factor)

    def saturation(self, img):
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(self.min_factor, self.max_factor)
        return enhancer.enhance(factor)

    def contrast(self, img):
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(self.min_factor, self.max_factor)
        return enhancer.enhance(factor)
    

class Flip(object):
    def __init__(self, prob=0.5, mode=None):
        """
        Args:
            prob(float): 翻转的可能性
            mode(str): h为水平翻转，v为垂直翻转，默认None是在h和v中随机选择
        """
        self.prob = prob
        self.mode = mode

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            if self.mode == 'h':
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.mode == 'v':
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            elif self.mode is None:
                if random.uniform(0, 1) < 0.5:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    return img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise ValueError(f"mode must be in ['v', 'h']")
        return img
    

class Resize(object):
    def __init__(self, size_in_pixel=None, size_in_scale=None):
        """
        Args:
            size_in_pixel: tuple (width, height)
            size_in_scale: tuple (width_scale, height_scale)
        """
        self.size_in_pixel = size_in_pixel
        self.size_in_scale = size_in_scale

    def __call__(self, img):
        if self.size_in_pixel is not None:
            return img.resize(self.size_in_pixel, Image.ANTIALIAS)
        elif self.size_in_scale is not None:
            current_size = img.size
            new_size = (int(current_size[0] * self.size_in_scale[0]), int(current_size[1] * self.size_in_scale[1]))
            return img.resize(new_size, Image.ANTIALIAS)
        else:
            print('size_in_pixel and size_in_scale are both None.')
            return img


class Target:
    def __init__(self, img, label, bbox=None):
        """
        Args:
            img(PIL.Image): 子图。
            label(str): 标签名。
            bbox(Tuple(xmin,ymin,xmax,ymax)): 在背景图上的坐标。
        """
        self.img = img
        self.label = label
        self.bbox = bbox


class SlideImg:
    def __init__(self, img, img_name, objects):
        """
        Args:
            img(PIL.Image): 拼接前的图。
            image_name(str): 拼接前，图的文件名。
            objects(List[{
                'class': cls_name,
                'bbox': (xmin, ymin, xmax, ymax)
            }, ...]): 小图上的目标
        """
        self.img = img
        self.img_name = img_name
        self.objects = objects
