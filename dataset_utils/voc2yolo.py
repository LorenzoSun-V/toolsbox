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
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import shutil
from utils import load_classes


def convert(size, box):
    """
    将绝对坐标转换为YOLO格式的相对坐标
    size: (width, height)
    box: (xmin, xmax, ymin, ymax)
    返回: (center_x, center_y, width, height) 归一化到 [0,1]
    """
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"图像尺寸无效: {size}")
    
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # 计算中心点坐标
    center_x = (box[0] + box[1]) / 2.0
    center_y = (box[2] + box[3]) / 2.0
    
    # 计算宽度和高度
    width = box[1] - box[0]
    height = box[3] - box[2]
    
    # 归一化
    center_x = center_x * dw
    center_y = center_y * dh
    width = width * dw
    height = height * dh
    
    # 确保值在 [0,1] 范围内
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return (center_x, center_y, width, height)


def convert_annotation(image_id, classes):
    try:
        with open('%s.xml' % (image_id), encoding="utf8", errors='ignore') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        
        # 检查 size 元素
        size = root.find('size')
        if size is None:
            print(f"警告: {image_id}.xml 中缺少 size 元素")
            return
            
        width_elem = size.find('width')
        height_elem = size.find('height')
        
        if width_elem is None or height_elem is None:
            print(f"警告: {image_id}.xml 中缺少 width 或 height 元素")
            return
            
        try:
            w = int(width_elem.text)
            h = int(height_elem.text)
        except (ValueError, TypeError):
            print(f"警告: {image_id}.xml 中图像尺寸无法转换为整数")
            return
    
        if w <= 0 or h <= 0:
            print(f"警告: {image_id}.xml 中图像尺寸无效: ({w}, {h})")
            return
        
        annotations = []  # 存储有效的标注
        
        for obj in root.iter('object'):
            try:
                # 处理 difficult 属性
                difficult = obj.find('difficult')
                if difficult is None:
                    difficult = obj.find('Difficult')
                difficult_value = '0' if difficult is None else difficult.text
                
                # 获取类别名称
                name_elem = obj.find('name')
                if name_elem is None:
                    print(f"警告: {image_id}.xml 中某个 object 缺少 name 元素")
                    continue
                    
                cls = name_elem.text
                if cls not in classes or int(difficult_value) == 1:
                    continue
                
                cls_id = classes.index(cls)
                
                # 获取边界框信息
                xmlbox = obj.find('bndbox')
                assert xmlbox is not None, f"警告: {image_id}.xml 中某个 object 缺少 bndbox 元素"
                
                # 检查边界框的各个坐标元素
                xmin_elem = xmlbox.find('xmin')
                ymin_elem = xmlbox.find('ymin')
                xmax_elem = xmlbox.find('xmax')
                ymax_elem = xmlbox.find('ymax')
                
                assert all(elem is not None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]), f"警告: {image_id}.xml 中某个 bndbox 缺少坐标元素"
                
                # 尝试转换坐标为浮点数
                try:
                    xmin = float(xmin_elem.text)
                    ymin = float(ymin_elem.text)
                    xmax = float(xmax_elem.text)
                    ymax = float(ymax_elem.text)
                except (ValueError, TypeError) as e:
                    print(f"警告: {image_id}.xml 中坐标值无法转换为数字: {e}")
                    continue
                
                # 验证边界框的有效性
                if xmin >= xmax or ymin >= ymax:
                    print(f"警告: {image_id}.xml 中边界框坐标无效: ({xmin}, {ymin}, {xmax}, {ymax})")
                    continue
                
                # 确保坐标在图像范围内
                if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                    print(f"警告: {image_id}.xml 中边界框超出图像范围: ({xmin}, {ymin}, {xmax}, {ymax}), 图像尺寸: ({w}, {h})")
                    # 裁剪到图像范围内
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmax)
                    ymax = min(h, ymax)
                    
                    # 重新检查裁剪后的边界框
                    if xmin >= xmax or ymin >= ymax:
                        print(f"警告: 裁剪后的边界框仍然无效，跳过")
                        continue
                
                # 转换为YOLO格式 (xmin, xmax, ymin, ymax)
                b = (xmin, xmax, ymin, ymax)
                
                try:
                    bb = convert((w, h), b)
                except ValueError as e:
                    print(f"警告: 坐标转换失败: {e}")
                    continue
                
                # 检查转换后的值是否有效
                if any(coord < 0 or coord > 1 for coord in bb):
                    print(f"警告: 转换后的坐标超出范围 [0,1]: {bb}")
                    continue
                
                annotations.append(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
                
            except Exception as e:
                print(f"处理 {image_id}.xml 中的对象时出错: {e}")
                continue
        
        # 写入文件
        if annotations:  # 只有当有有效标注时才创建文件
            with open('%s.txt' % (image_id), 'w') as out_file:
                out_file.writelines(annotations)
        # else:
        #     print(f"警告: {image_id}.xml 中没有有效的标注")
            
    except ET.ParseError as e:
        print(f"XML解析错误: {image_id}.xml - {e}")
        return
    except FileNotFoundError:
        print(f"文件不存在: {image_id}.xml")
        return
    except Exception as e:
        print(f"处理 {image_id}.xml 时发生未知错误: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="create classes")
    parser.add_argument('--voc-label-list', help='In Voc format dataset, path to label list. The content of each line is a category.', type=Path, default=None)
    parser.add_argument('--xml-dir', type=str, help='which classes to do xml2yolo')
    args = parser.parse_args()
    
    try:
        classes = load_classes(args.voc_label_list)
    except Exception as e:
        print(f"加载类别列表时出错: {e}")
        return

    root_path = args.xml_dir
    xml_dirs = glob(join(root_path, "*.xml"))
    
    if not xml_dirs:
        print(f"在 {root_path} 中未找到 XML 文件")
        return
    
    print(f"开始处理 {len(xml_dirs)} 个 XML 文件...")
    for xml_dir in tqdm(xml_dirs):
        convert_annotation(xml_dir[:-4], classes)

    # 移动生成的 txt 文件
    root_path_parent = os.path.dirname(args.xml_dir.rstrip(os.path.sep)) if args.xml_dir.endswith('/') else os.path.dirname(args.xml_dir)
    save_dir = join(root_path_parent, "labels")
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir, exist_ok=True)

    try:
        txt_files = [f for f in os.listdir(args.xml_dir) if f.endswith('.txt')]
        print(f"移动 {len(txt_files)} 个 txt 文件到 {save_dir}")
        
        for txt_file in tqdm(txt_files):
            try:
                source_path = os.path.join(args.xml_dir, txt_file)
                dest_path = os.path.join(save_dir, txt_file)
                shutil.move(source_path, dest_path)
            except Exception as e:
                print(f"移动文件 {txt_file} 时出错: {e}")
                
    except Exception as e:
        print(f"处理 txt 文件时出错: {e}")


if __name__ == '__main__':
    main()