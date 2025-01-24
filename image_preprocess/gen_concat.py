import os
import os.path as osp
import xml.etree.ElementTree as ET
import time
from PIL import Image
from glob import glob
from xml.dom import minidom
from tqdm import tqdm


def generate_labels(window_folder, voc_labels_folder, output_folder, n):
    window_paths = sorted(glob(osp.join(window_folder, "*.jpg")))
    image_count = 0

    for i, window_path in tqdm(enumerate(window_paths)):
        joint_begin_exists = False
        joint_end_exists = False

        joint_begin_path = os.path.join(voc_labels_folder, os.path.basename(window_path).replace(".jpg", ".xml"))
        joint_end_path = os.path.join(voc_labels_folder, os.path.basename(window_path).replace(".jpg", ".xml"))
        if os.path.exists(joint_begin_path) and os.path.exists(joint_end_path):
            joint_begin_exists = True
            joint_end_exists = True
        
        if joint_begin_exists and joint_end_exists:
            # Parse coordinates from joint_begin and joint_end files
            joint_begin_coords = parse_voc_xml(joint_begin_path)
            joint_end_coords = parse_voc_xml(joint_end_path)
            
            # Calculate joint bbox coordinates
            xmin1, ymin1, xmax1, ymax1 = joint_begin_coords
            xmin2, ymin2, xmax2, ymax2 = joint_end_coords
            
            # Write joint label to output VOC file
            output_image_name = f"{int(time.time() * 1000):13d}_{image_count}.jpg"
            output_label_name = f"{int(time.time() * 1000):13d}_{image_count}.xml"
            output_image_path = os.path.join(output_folder, output_image_name)
            output_label_path = os.path.join(output_folder, output_label_name)
            
            image_count += 1
            
            # Concatenate images
            new_img = concatenate_images(window_paths[i:i+n], output_image_path)
            w, h = new_img.size
            
            write_voc_xml(output_label_path, output_image_path, xmin1, ymin2, xmax1, ymax1, w, h)
            
            # Reset flags for next window
            joint_begin_exists = False
            joint_end_exists = False
            # continue
        else:
            # Skip generating labels for this window if joint_begin and joint_end not both exist
            continue

def concatenate_images(image_paths, output_path):
    images = [Image.open(image_path) for image_path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width
    new_image.save(output_path)
    return new_image

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        return (xmin, ymin, xmax, ymax)

def write_voc_xml(xml_file, jpg_file, xmin, ymin, xmax, ymax, w, h):
    annotation = ET.Element("annotation")

    # 添加图像文件夹和文件名
    ET.SubElement(annotation, "folder").text = osp.dirname(jpg_file)
    ET.SubElement(annotation, "filename").text = osp.basename(jpg_file)

    # 添加图像尺寸信息
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(3)

    object_elem = ET.SubElement(annotation, "object")
    bndbox = ET.SubElement(object_elem, "bndbox")
    ET.SubElement(object_elem, "name").text = "joint"
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)
    
    # tree = ET.ElementTree(annotation)
    # tree.write(xml_file)
    # 生成格式化的XML字符串
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    
        # 将格式化的XML内容写入文件
    with open(xml_file, "w") as f:
        f.write(xmlstr)


# Example usage:
window_folder = "/data/bt/xray_fanglun/LabeledData/20240125_all/images"
voc_labels_folder = "/data/bt/xray_fanglun/LabeledData/20240125_all/voc_labels"
output_folder = "/data/bt/xray_fanglun/LabeledData/20240125_concat"
n = 10  # Number of windows to concatenate
generate_labels(window_folder, voc_labels_folder, output_folder, n)
