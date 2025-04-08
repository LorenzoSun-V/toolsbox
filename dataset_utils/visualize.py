'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-08 01:28:14
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2024-01-17 07:58:16
Description: 可视化工具，用于可视化cocostyle/yolostyle/vocstyle数据集
'''
from pathlib import Path
import cv2
import json
import os.path as osp
import numpy as np
import shutil
from tqdm import tqdm
from glob import glob
import xml.etree.ElementTree as ET


class Visualizer:
    def __init__(
        self,
        img_dir: str,
        vis_dir: str,
        ) -> None:
        """
        Args:
            img_dir (str): 图片文件夹路径
            vis_dir (str): 可视化结果保存路径
        """
        self.img_dir = img_dir
        self.vis_dir = vis_dir
        self.pattle = [(0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
                        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
                        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
                        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
                        (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
                        (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
                        (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
                        (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
                        (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
                        (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
                        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
                        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
                        (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
                        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
                        (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
                        (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
                        (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
                        (246, 0, 122), (191, 162, 208)]

    def vis_coco_style(self, json_file: str) -> None:
        """
        该函数用于可视化coco格式的数据集，可视化结果保存在vis_dir/coco文件夹下
        Args:
            json_file (str): coco格式的json文件路径
        """
        (Path(self.vis_dir) / "coco").mkdir(parents=True, exist_ok=True)
        data = json.load(open(json_file, 'r'))
        id_category = {}
        for category in data["categories"]:
            id_category[category["id"]] = category["name"]
        images = data['images']
        for i in tqdm(images):
            img = cv2.imread(osp.join(self.img_dir, i['file_name']))
            bboxes = []
            category_ids = []
            annotations = data['annotations']
            for j in annotations:
                if j['image_id'] == i['id']:
                    bboxes.append(j["bbox"])
                    category_ids.append(j['category_id'])
    
            # 只查看包含某一类别的图片
            # if 6 not in category_ids:
            #     continue
    
            # 生成锚框
            for idx, bbox in enumerate(bboxes):
                left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
                right_bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
                cv2.rectangle(img, left_top, right_bottom, self.pattle[category_ids[idx]], 2)  # 图像，左上角，右下坐标，颜色，粗细
                cv2.putText(img, f"{category_ids[idx]}-{id_category[category_ids[idx]]}", left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self.pattle[category_ids[idx]], 2)
                # 画出每个bbox的类别，参数分别是：图片，类别名(str)，坐标，字体，大小，颜色，粗细
            cv2.imwrite(osp.join(self.vis_dir, "coco", i['file_name']), img)

    def vis_yolo_style(self, label_folder: str) -> None:
        """
        该函数用于可视化yolo格式的数据集，可视化结果保存在vis_dir/yolo文件夹下
        Args:
            label_folder (str): yolo格式的label文件夹路径
        """
        (Path(self.vis_dir) / "yolo").mkdir(parents=True, exist_ok=True)
        img_list = sorted(glob(osp.join(self.img_dir, "*.[jJ][pP][gG]*"))+glob(osp.join(self.img_dir, "*.[jJ][pP][eE][gG]*"))+glob(osp.join(self.img_dir, "*.[pP][nN][gG]*")))

        for image_path in tqdm(img_list):
            file_name = Path(image_path).stem
            label_path = Path(label_folder) / f"{file_name}.txt"
            if not label_path.is_file(): 
                (Path(self.vis_dir) / "yolo" / "neg_samples").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, str((Path(self.vis_dir) / "yolo" / "neg_samples" / f"{file_name}.jpg")))
                continue
            # 读取图像文件
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            # 读取 labels
            with open(label_path, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                for x in lb:
                    img = self.xywh2xyxy(x, w, h, img)
            cv2.imwrite(osp.join(self.vis_dir, "yolo", f"{file_name}.jpg"), img)
    
    def vis_xyxy_style(self, label_folder: str) -> None:
        """
        该函数用于可视化xyxy格式的数据集，可视化结果保存在vis_dir/xyxy文件夹下
        Args:
            label_folder (str): xyxy格式的label文件夹路径
        """
        (Path(self.vis_dir) / "xyxy").mkdir(parents=True, exist_ok=True)
        img_list = sorted(glob(osp.join(self.img_dir, "*.[jJ][pP][gG]*"))+glob(osp.join(self.img_dir, "*.[jJ][pP][eE][gG]*"))+glob(osp.join(self.img_dir, "*.[pP][nN][gG]*")))

        for image_path in tqdm(img_list):
            file_name = Path(image_path).stem
            label_path = Path(label_folder) / f"{file_name}.txt"
            if not label_path.is_file(): 
                (Path(self.vis_dir) / "xyxy" / "neg_samples").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, str((Path(self.vis_dir) / "xyxy" / "neg_samples" / f"{file_name}.jpg")))
                continue
            # 读取图像文件
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            # 读取 labels
            with open(label_path, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                for x in lb:
                    if len(x) == 6:
                        # label, top_left_x, top_left_y, bottom_right_x, bottom_right_y, _ = [int(item) for item in x]
                        top_left_x, top_left_y, bottom_right_x, bottom_right_y, _, label = [int(item) for item in x]
                    elif len(x) == 5:
                        label, top_left_x, top_left_y, bottom_right_x, bottom_right_y = [int(item) for item in x]
                    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.pattle[label], 2)
                    cv2.putText(img, str(label), (top_left_x, top_left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.pattle[label], 2)

            cv2.imwrite(osp.join(self.vis_dir, "xyxy", f"{file_name}.jpg"), img)
    
    def xywh2xyxy(self, x, w1, h1, img):
        label, x, y, w, h = x
        label = int(label)
        # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
        # 边界框反归一化
        x_t = x * w1
        y_t = y * h1
        w_t = w * w1
        h_t = h * h1
        # print("反归一化后输出：\n第一个:{}\t第二个:{}\t第三个:{}\t第四个:{}\t\n\n".format(x_t, y_t, w_t, h_t))
        # 计算坐标
        top_left_x = int(x_t - w_t / 2)
        top_left_y = int(y_t - h_t / 2)
        bottom_right_x = int(x_t + w_t / 2)
        bottom_right_y = int(y_t + h_t / 2)

        # 绘制矩形框
        # cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,0,255), 2)
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), self.pattle[label], 2)
        cv2.putText(img, str(label), (top_left_x, top_left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.pattle[label], 2)
        """
        # (可选)给不同目标绘制不同的颜色框
        if int(label) == 0:
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
        elif int(label) == 1:
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (255, 0, 0), 2)
        """
        return img
    
    def vis_voc_style(self, annot_folder: str) -> None:
        """
        该函数用于可视化VOC格式的数据集，可视化结果保存在vis_dir/voc文件夹下
        Args:
            annot_folder (str): VOC格式的annotation文件夹路径（XML文件）
        """
        (Path(self.vis_dir) / "voc").mkdir(parents=True, exist_ok=True)
        img_list = sorted(glob(osp.join(self.img_dir, "*.[jJ][pP][gG]*"))+glob(osp.join(self.img_dir, "*.[jJ][pP][eE][gG]*"))+glob(osp.join(self.img_dir, "*.[pP][nN][gG]*")))

        for image_path in tqdm(img_list):
            file_name = Path(image_path).stem
            annot_path = Path(annot_folder) / f"{file_name}.xml"
            if not annot_path.is_file():
                (Path(self.vis_dir) / "voc" / "neg_samples").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, str((Path(self.vis_dir) / "voc" / "neg_samples" / f"{file_name}.jpg")))
                continue

            # 读取图像文件
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]

            # 解析VOC XML文件
            tree = ET.parse(str(annot_path))
            root = tree.getroot()
            objects = root.findall('object')

            for obj in objects:
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                # 获取类别的颜色索引（根据需求可以进行扩展）
                label = name
                label_idx = self._get_label_index(label)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.pattle[label_idx], 2)
                cv2.putText(img, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.pattle[label_idx], 2)

            # 保存可视化结果
            cv2.imwrite(osp.join(self.vis_dir, "voc", f"{file_name}.jpg"), img)

    def _get_label_index(self, label: str) -> int:
        """
        Helper function to map the label name to an index for color palette.
        Args:
            label (str): The object label name
        Returns:
            int: The index for the color palette
        """
        # In this example, the labels are mapped to colors via index.
        # Modify this function to fit the actual label to index mapping.
        # For simplicity, assume that there are 20 classes.
        label_list = ["person", "headshoulder", ""]
        if label in label_list:
            return label_list.index(label)
        else:
            assert False, "Label not found in the label list"
            return 0  # Default to class 0 if label not found