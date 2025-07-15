import os
import shutil
import xml.etree.ElementTree as ET
import argparse
import json


def parse_xml_annotation(xml_path):
    """解析VOC格式的XML标注文件，返回所有类别"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        classes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.append(class_name)
        
        return classes
    except:
        return []


def create_dataset_split(source_dir, dataset_configs):
    """
    创建数据集拆分
    
    Args:
        source_dir: 源数据集目录
        dataset_configs: 数据集配置字典，格式为 {子集名称: [类别列表]}
    """
    
    images_dir = os.path.join(source_dir, 'images')
    annotations_dir = os.path.join(source_dir, 'Annotations')
    
    # 检查源目录是否存在
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"错误: 找不到源目录 {images_dir} 或 {annotations_dir}")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    for subset_name, target_classes in dataset_configs.items():
        print(f"\n开始处理数据集: {subset_name}")
        print(f"目标类别: {target_classes}")
        
        # 创建子数据集目录结构
        subset_dir = os.path.join(source_dir, 'subdataset', subset_name)
        subset_images_dir = os.path.join(subset_dir, 'images')
        subset_annotations_dir = os.path.join(subset_dir, 'Annotations')
        subset_trainval_dir = os.path.join(subset_dir, 'trainval')
        
        os.makedirs(subset_images_dir, exist_ok=True)
        os.makedirs(subset_annotations_dir, exist_ok=True)
        os.makedirs(subset_trainval_dir, exist_ok=True)
        
        # 筛选符合条件的图片
        valid_images = []
        
        for image_file in image_files:
            # 获取对应的XML文件名
            base_name = os.path.splitext(image_file)[0]
            xml_file = base_name + '.xml'
            xml_path = os.path.join(annotations_dir, xml_file)
            
            # 如果XML文件不存在，跳过
            if not os.path.exists(xml_path):
                continue
            
            # 解析XML获取类别
            classes_in_image = parse_xml_annotation(xml_path)
            
            # 检查是否包含目标类别或为空标签
            if not classes_in_image or any(cls in target_classes for cls in classes_in_image):
                valid_images.append(image_file)
                
                # 直接复制图片
                src_image_path = os.path.join(images_dir, image_file)
                dst_image_path = os.path.join(subset_images_dir, image_file)
                shutil.copy2(src_image_path, dst_image_path)
                
                # 直接复制XML标注文件（不修改内容）
                src_xml_path = os.path.join(annotations_dir, xml_file)
                dst_xml_path = os.path.join(subset_annotations_dir, xml_file)
                shutil.copy2(src_xml_path, dst_xml_path)
        
        print(f"筛选出 {len(valid_images)} 张图片")
        
        # 生成classes.txt文件
        if target_classes:
            classes_txt_path = os.path.join(subset_trainval_dir, 'classes.txt')
            with open(classes_txt_path, 'w') as f:
                for cls in target_classes:
                    f.write(cls + '\n')
            
            print(f"数据集 {subset_name} 创建完成!")
            print(f"包含图片: {len(valid_images)} 张")
            print(f"类别文件已生成: {classes_txt_path}")
        else:
            print(f"警告: 数据集 {subset_name} 没有定义类别")


def load_config_from_file(config_file):
    """从JSON文件加载数据集配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_file} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式错误 - {e}")
        return None


def parse_config_from_string(config_string):
    """从JSON字符串解析数据集配置"""
    try:
        config = json.loads(config_string)
        return config
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式错误 - {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='拆分目标检测数据集')
    parser.add_argument('--source_dir', '-s', type=str, required=True,
                        help='源数据集目录 (默认: 当前目录)')
    parser.add_argument('--config', '-c', type=str, 
                        help='数据集配置JSON字符串')
    parser.add_argument('--config_file', '-f', type=str,
                        help='数据集配置JSON文件路径')
    
    args = parser.parse_args()
    
    # 读取数据集配置
    dataset_configs = None
    
    if args.config_file:
        # 从文件读取配置
        dataset_configs = load_config_from_file(args.config_file)
    elif args.config:
        # 从命令行参数读取配置
        dataset_configs = parse_config_from_string(args.config)
    else:
        # 使用默认配置
        dataset_configs = {
            'fire-smoke': ['fire', 'smoke'],
            'person': ['person'],
            'person_behavior': ['nomask', 'smoking', 'nowf', 'fall', 'sleep', 'tx', 'wsf', 'mask', 'wf', 'legs']
        }
        print(f"使用默认配置: {dataset_configs}")
    
    if dataset_configs is None:
        print("错误: 无法加载数据集配置")
        return
    
    print("开始拆分数据集...")
    print(f"源目录: {args.source_dir}")
    print(f"数据集配置: {json.dumps(dataset_configs, ensure_ascii=False, indent=2)}")
    
    # 执行数据集拆分
    create_dataset_split(args.source_dir, dataset_configs)
    
    print("\n数据集拆分完成!")

if __name__ == "__main__":
    main()