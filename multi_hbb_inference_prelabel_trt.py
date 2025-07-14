import sys
import yaml
import cv2
import os
import os.path as osp
from inference.yolo import YOLODetector
from inference.utils import detbox_to_shape_rectangle, save_anylabeling_json
from tqdm import tqdm


# --------- Config Parsing --------- #
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def init_models(cfg):
    models = []
    for model_cfg in cfg['models']:
        model_path = model_cfg['model_path']
        model_type = model_cfg['model_type']
        assert model_type in ['yoloe2e', 'yolov8'], f"{model_type} is not supported now, only support hbb model yoloe2e&yolov8 now"
        class_names = model_cfg['class_names']
        assert osp.isfile(model_path), f"Model file {model_path} does not exist"
        model = YOLODetector(
            model_path=model_path,
            model_type=model_type,
            class_names=class_names,
            conf_threshold=model_cfg['conf_threshold'],
            nms_threshold=model_cfg['nms_threshold']
        )
        models.append(model)
    return models


def load_image_paths(img_dir):
    batch_paths = []
    print(f"Total images to process: {len(os.listdir(img_dir))}")
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.endswith(".jpg") and not img_name.endswith(".png") and \
           not img_name.endswith(".jpeg") and not img_name.endswith(".JPG") and \
           not img_name.endswith(".JPEG") and not img_name.endswith(".PNG"):
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

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        shapes = []
        for model_idx, model in enumerate(models):
            results = model.detect(img)
            for obj in results:
                class_id = obj[-1]
                if model.class_names[class_id] not in cfg['models'][model_idx]['class_filter']:
                    continue
                shape_item = detbox_to_shape_rectangle(obj, model.class_names)
                shapes.append(shape_item)
        save_anylabeling_json(img_path, shapes, cfg['input_img_dir'], img.shape)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trt_multi_model_class_specific_prelabel.py config.yaml")
        exit(1)
    main(sys.argv[1])