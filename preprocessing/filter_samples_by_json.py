import os
import json
import shutil
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def select_target(folder_path, cls_name, num=1):
    """
    根据类名和数量选择目标图片
    
    Args:
        folder_path: 包含json文件的文件夹路径
        cls_name: 目标类名列表
        num: 每个类的最小数量阈值
    """
    # 获取所有json文件
    json_files = sorted(glob(os.path.join(folder_path, "*.json")))
    
    if not json_files:
        print("未找到任何JSON文件")
        return
    
    # 创建目标目录
    target_dir = os.path.join(os.path.dirname(folder_path), "select")
    os.makedirs(target_dir, exist_ok=True)
    
    # 支持的图片扩展名
    supported_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']
    
    # 转换为集合以提高查找效率
    cls_name_set = set(cls_name)
    
    selected_count = 0
    
    for json_file in tqdm(json_files, desc="处理文件"):
        # 查找对应的图片文件
        img_file = find_corresponding_image(json_file, supported_extensions)
        
        if not img_file:
            print(f"警告: 未找到 {json_file} 对应的图片文件")
            continue
        
        # 读取并解析JSON文件
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"错误: 无法读取 {json_file}: {e}")
            continue
        
        # 统计当前文件中各类的数量
        label_counts = count_labels(data, cls_name_set)
        
        # 检查是否满足条件
        if any(count >= num for count in label_counts.values()):
            try:
                # 复制文件到目标目录
                shutil.copy2(json_file, target_dir)
                shutil.copy2(img_file, target_dir)
                selected_count += 1
                
                # 打印当前文件的标签统计
                label_info = ", ".join([f"{label}: {count}" for label, count in label_counts.items() if count > 0])
                print(f"已选择: {os.path.basename(json_file)} ({label_info})")
                
            except Exception as e:
                print(f"错误: 复制文件失败 {json_file}: {e}")
    
    print(f"完成! 共选择了 {selected_count} 个文件到 {target_dir}")


def find_corresponding_image(json_file, extensions):
    """查找对应的图片文件"""
    base_name = os.path.splitext(json_file)[0]
    
    for ext in extensions:
        img_file = base_name + ext
        if os.path.exists(img_file):
            return img_file
    
    return None


def count_labels(data, target_labels):
    """统计标签数量"""
    label_counts = defaultdict(int)
    
    # 初始化目标标签计数
    for label in target_labels:
        label_counts[label] = 0
    
    # 统计shapes中的标签
    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label')
        if label in target_labels:
            label_counts[label] += 1
    
    return dict(label_counts)


# 使用示例
if __name__ == "__main__":
    folder_path = "/data/nofar/person_behavior/labeled_data/labeling_rule_v2.0.0/20250702/images"
    cls_name = ["sleep", "fall"]
    select_target(folder_path, cls_name, 1)