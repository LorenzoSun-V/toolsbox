'''
Author: BTZN0325 sunjiahui@boton-tech.com
Date: 2024-01-04 15:49:32
LastEditors: BTZN0325 sunjiahui@boton-tech.com
LastEditTime: 2024-01-30 08:00:45
Description: 图像拼接，拼接接头开始、接头中间、接头结尾。目前代码写得有点乱，后续优化。
'''
import os
import os.path as osp
from PIL import Image
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from time import time
from utils import iou, SlideImg


def create_voc_xml(objects, out_dir, filename, img_width, img_height, img_depth=3):
    # 创建根元素
    annotation = ET.Element("annotation")

    # 添加图像文件夹和文件名
    ET.SubElement(annotation, "folder").text = out_dir
    ET.SubElement(annotation, "filename").text = filename

    # 添加图像尺寸信息
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    # 为列表中的每个对象添加object元素
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj['class']
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj['bbox'][0])
        ET.SubElement(bndbox, "ymin").text = str(obj['bbox'][1])
        ET.SubElement(bndbox, "xmax").text = str(obj['bbox'][2])
        ET.SubElement(bndbox, "ymax").text = str(obj['bbox'][3])

    # 生成格式化的XML字符串
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    
        # 将格式化的XML内容写入文件
    with open(os.path.join(out_dir, filename.rsplit('.', 1)[0] + '.xml'), "w") as f:
        f.write(xmlstr)

    print(f"VOC label for {filename} generated in {out_dir}")


def parse_voc_annotation(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    objects = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(float(xml_box.find('xmin').text))
        ymin = int(float(xml_box.find('ymin').text))
        xmax = int(float(xml_box.find('xmax').text))
        ymax = int(float(xml_box.find('ymax').text))
        objects.append({'class': cls_name, 'bbox': (xmin, ymin, xmax, ymax)})

    return objects


def group_images_by_time(slide_img_list):
    grouped_images = {}
    for slide_img in slide_img_list:
        # 提取时间戳 "hh-mm"
        time_stamp = slide_img.img_name.split('-')[:2]
        time_key = '-'.join(time_stamp)

        # 如果字典中不存在这个键，则创建一个新列表
        if time_key not in grouped_images:
            grouped_images[time_key] = []

        # 将SlideImg对象添加到相应的列表中
        grouped_images[time_key].append(slide_img)

    # 返回所有分组后的列表
    return list(grouped_images.values())


def stitch_images(images, direction='horizontal'):
    """
    Stitch a list of PIL.Image objects either horizontally or vertically.
    
    :param images: List of PIL.Image objects to be stitched.
    :param direction: 'horizontal' or 'vertical' for the stitching direction.
    :return: A new PIL.Image object with the stitched images.
    """

    if not images:
        return None

    # Check image dimensions
    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        if len(set(heights)) > 1:  # Check if all heights are the same
            raise ValueError("All images must have the same height for horizontal stitching.")
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.width

    elif direction == 'vertical':
        if len(set(widths)) > 1:  # Check if all widths are the same
            raise ValueError("All images must have the same width for vertical stitching.")
        total_height = sum(heights)
        max_width = max(widths)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.height

    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    return new_im


def load_images(folder_path):
    """
    Args:
        folder_path(str): 存放素材图的文件夹，下面有 images 和 voc_labels 文件夹分别存放图片和对应的标签。
    Return:
        img_list(List(SlideImg)): 
    """
    image_path = osp.join(folder_path, "images")
    label_path = osp.join(folder_path, "voc_labels")
    assert osp.exists(image_path), f"{image_path} is not existed!"
    assert osp.exists(label_path), f"{label_path} is not existed!"

    img_list = []
    # 为确保图片顺序，因此按文件名称排序
    for image_name in sorted(os.listdir(image_path)):
        img = Image.open(osp.join(image_path, image_name))
        label_dir = osp.join(label_path, image_name.replace('jpg', 'xml'))
        if not osp.exists(label_dir):
            objects = []
        else:
            objects = parse_voc_annotation(label_dir)
        img_list.append(SlideImg(img, image_name, objects))

    return img_list


if __name__ == "__main__":
    num = 2000  # 随机生成图片的数量
    base_h = 1920
    base_w = 1024
    out_dir = "/data/bt/xray_fanglun/cls2_20240111_v0.1/images"

    root_dir = "/data/bt/xray_fanglun/material/20240110"
    joint_begin_root = osp.join(root_dir, "joint_begin")
    joint_mid_root = osp.join(root_dir, "joint_mid")
    joint_end_root = osp.join(root_dir, "joint_end")
    # 按照文件名前5位分组，因为有的接头开始和结尾部分，会被拆成两张图像
    # 一般图像名是 "hh-mm-ss-number.jpg" 这种格式的
    joint_begin_group = group_images_by_time(load_images(joint_begin_root))
    joint_mid_group = group_images_by_time(load_images(joint_mid_root))
    joint_end_group = group_images_by_time(load_images(joint_end_root))


    img_dirs = []
    # 制作完整接头
    for gen_num in range(num):
        new_objects = []
        joints = []
        begin_key = random.choice(list(joint_begin_group.keys()))
        mid_key = random.choice(list(joint_mid_group.keys()))
        end_key = random.choice(list(joint_end_group.keys()))
        num_stitch_images = 0  # 拼接图像计数，用于计算坐标
        num_stitch_images_max = len(joint_begin_group[begin_key]) + len(joint_mid_group[mid_key]) + len(joint_end_group[end_key]) - 1

        # 拼接每组图像
        begin_image = stitch_images([Image.open(osp.join(joint_begin_root, "images", f"{name}.jpg")) for name in joint_begin_group[begin_key]])
        for name in joint_begin_group[begin_key]:
            voc_label_path = osp.join(joint_begin_root, "voc_labels", f"{name}.xml")
            # 有标签说明有目标，需要计算目标在拼接图上的坐标
            if osp.exists(voc_label_path):
                objects = parse_voc_annotation(voc_label_path)
                for object in objects:
                    # 只要第一张图的joint_begin
                    if object['class'] == "joint_begin" and num_stitch_images != 0:
                        continue
                    xmin, ymin, xmax, ymax = object['bbox']
                    # x方向要按照拼接图的数量累加
                    xmin += base_w*num_stitch_images
                    xmax += base_w*num_stitch_images
                    object['bbox'] = (xmin, ymin, xmax, ymax)
                    # 接头目标需要单独处理，拼接 joint_begin和joint_end
                    if object['class'] == "joint_begin":
                        joints.append(object['bbox'])
                        continue
                    new_objects.append(object)
            num_stitch_images += 1

        mid_image = stitch_images([Image.open(osp.join(joint_mid_root, "images", f"{name}.jpg")) for name in joint_mid_group[mid_key]])
        for name in joint_mid_group[mid_key]:
            voc_label_path = osp.join(joint_mid_root, "voc_labels", f"{name}.xml")
            if osp.exists(voc_label_path):
                objects = parse_voc_annotation(voc_label_path)
                for object in objects:
                    xmin, ymin, xmax, ymax = object['bbox']
                    xmin += base_w*num_stitch_images
                    xmax += base_w*num_stitch_images
                    object['bbox'] = (xmin, ymin, xmax, ymax)
                    new_objects.append(object)
            num_stitch_images += 1

        end_image = stitch_images([Image.open(osp.join(joint_end_root, "images", f"{name}.jpg")) for name in joint_end_group[end_key]])
        for name in joint_end_group[end_key]:
            voc_label_path = osp.join(joint_end_root, "voc_labels", f"{name}.xml")
            # 有标签说明有目标，需要计算目标在拼接图上的坐标
            if osp.exists(voc_label_path):
                objects = parse_voc_annotation(voc_label_path)
                for object in objects:
                    # 只要最后一张图的joint_end
                    if object['class'] == "joint_end" and num_stitch_images != num_stitch_images_max:
                        continue
                    xmin, ymin, xmax, ymax = object['bbox']
                    # x方向要按照拼接图的数量累加
                    xmin += base_w*num_stitch_images
                    xmax += base_w*num_stitch_images
                    object['bbox'] = (xmin, ymin, xmax, ymax)
                    # 接头目标需要单独处理，拼接 joint_begin和joint_end
                    if object['class'] == "joint_end":
                        joints.append(object['bbox'])
                        continue
                    new_objects.append(object)
            num_stitch_images += 1

        gen_image = stitch_images([begin_image, mid_image, end_image])
        timestamp = int(time()*1000)
        filename = osp.join(out_dir, f"{timestamp:13d}_{gen_num}.jpg")
        gen_image.save(osp.join(out_dir, filename))
        img_dirs.append(osp.join(out_dir, filename))
        # 将 joint_begin和joint_end 处理成一个joint
        if len(joints) != 2: raise ValueError(f"num of joints must be 2!")
        new_objects.append({'class': 'joint', 'bbox': (joints[0][0], joints[1][1], joints[1][2], joints[0][3])})
        create_voc_xml(new_objects, out_dir, filename, base_w*num_stitch_images, base_h)

    # 在制作的完成接头上，随机crop制作非完整接头，高还是保持 base_h
    for img_dir in img_dirs:
        img = Image.open(img_dir)
        img_name = osp.basename(img_dir).rsplit('.', 1)[0]
        w, h = img.size
        random_x = random.randint(0, w-2*base_w)

        if random_x < base_w:
            random_w = random.randint(base_w*2, w-random_x-base_w)
        else:
            random_w = random.randint(base_w*2, w-random_x)
        crop_bbox = (random_x, 0, random_x+random_w, base_h)  # 相对于原图坐标系，crop图像的坐标
        crop_img = img.crop(crop_bbox)

        # 获取原图所有目标
        voc_label_path = img_dir.replace('jpg', 'xml')
        new_objects = []
        if osp.exists(voc_label_path):
            objects = parse_voc_annotation(voc_label_path)
        else:
            objects = []
        for object in objects:
            # 不需要 joint, 且目标必须完全在crop图像上
            if (object['class'] != "joint") and \
                (iou(object['bbox'], crop_bbox)[1] == (object['bbox'][2]-object['bbox'][0])*(object['bbox'][3]-object['bbox'][1])):
                # 将目标坐标从原图坐标系转换到crop图像坐标系上
                xmin, ymin, xmax, ymax = object['bbox']
                xmin -= random_x
                xmax -= random_x
                object['bbox'] = (xmin, ymin, xmax, ymax)
                new_objects.append(object)
        filename = osp.join(out_dir, f"{img_name}_crop.jpg")
        crop_img.save(filename)
        create_voc_xml(new_objects, out_dir, f"{img_name}_crop.jpg", crop_img.size[0], crop_img.size[1])