import sys
import yaml
import cv2
import os
import os.path as osp
from inference.yolo import YOLODetector
# 导入新的工具函数
from inference.utils import detbox_to_shape_rectangle, detbox_to_shape_rotation, save_anylabeling_json
from tqdm import tqdm


# --------- Config Parsing --------- #
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def init_models(cfg):
    models = []
    # 建立模型类型到处理函数的映射
    shape_converters = {
        'yoloe2e': detbox_to_shape_rectangle,
        'yolov8obb': detbox_to_shape_rotation
    }
    
    for model_cfg in cfg['models']:
        model_path = model_cfg['model_path']
        model_type = model_cfg['model_type']
        # 扩展支持的模型类型
        assert model_type in shape_converters, f"{model_type} is not supported. Supported types are {list(shape_converters.keys())}"
        class_names = model_cfg['class_names']
        assert osp.isfile(model_path), f"Model file {model_path} does not exist"
        
        model = YOLODetector(
            model_path=model_path,
            model_type=model_type,
            class_names=class_names,
            conf_threshold=model_cfg['conf_threshold'],
            nms_threshold=model_cfg['nms_threshold']
        )
        
        # 将模型和其对应的转换函数一起存储
        models.append({
            'instance': model,
            'converter': shape_converters[model_type],
            'class_filter': model_cfg.get('class_filter', []) # 使用 .get() 避免 class_filter 不存在时出错
        })
    return models


def load_image_paths(img_dir):
    batch_paths = []
    print(f"Total images to process: {len(os.listdir(img_dir))}")
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = osp.join(img_dir, img_name)
        batch_paths.append(img_path)

    return batch_paths


def main(cfg_path):
    assert osp.isfile(cfg_path), f"Config file {cfg_path} does not exist"
    cfg = load_config(cfg_path)
    assert osp.exists(cfg['input_img_dir']), f"Input image directory {cfg['input_img_dir']} does not exist"
    models = init_models(cfg)
    img_paths = load_image_paths(cfg['input_img_dir'])

    for img_path in tqdm(img_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image {img_path}, skipping.")
            continue
            
        shapes = []
        for model_info in models:
            model = model_info['instance']
            converter = model_info['converter']
            class_filter = model_info['class_filter']
            
            results = model.detect(img)
            
            for obj in results:
                class_id = int(obj[-1])
                # 如果定义了类别过滤器且当前类别不在其中，则跳过
                if class_filter and model.class_names[class_id] not in class_filter:
                    continue
                
                # 使用与模型类型匹配的转换函数
                print(obj)
                shape_item = converter(obj, model.class_names)
                
                shapes.append(shape_item)
                
        save_anylabeling_json(img_path, shapes, cfg['input_img_dir'], img.shape)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python multi_hbb_inference_prelabel_trt.py config.yaml")
        exit(1)
    main(sys.argv[1])