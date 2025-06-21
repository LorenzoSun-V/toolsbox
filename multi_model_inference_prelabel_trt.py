import sys
import yaml
import cv2
import os
import os.path as osp
from inference.yolo import Yolov8Detector
from inference.utils import detbox_to_shape_rectangle, save_anylabeling_json
from tqdm import tqdm


# --------- Config Parsing --------- #
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def init_models(cfg):
    lib_path = cfg['lib_paths']['yoloe2e']
    assert osp.isfile(lib_path), f"Library file {lib_path} does not exist"
    models = []
    for model_cfg in cfg['models']:
        model_path = model_cfg['model_path']
        assert osp.isfile(model_path), f"Model file {model_path} does not exist"
        model = Yolov8Detector(
            library_path=lib_path,
            weights_path=model_path,
            conf_threshold=model_cfg['conf_threshold'],
            nms_threshold=model_cfg['nms_threshold']
        )
        models.append(model)
    return models


def load_images(img_dir):
    batch_frames = []
    batch_paths = []
    print(f"Total images to process: {len(os.listdir(img_dir))}")
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.endswith(".jpg") and not img_name.endswith(".png") and \
           not img_name.endswith(".jpeg") and not img_name.endswith(".JPG") and \
           not img_name.endswith(".JPEG") and not img_name.endswith(".PNG"):
            continue
        img_path = osp.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        batch_frames.append(img)
        batch_paths.append(img_path)
    print(f"Total images loaded: {len(batch_frames)}")
    return batch_frames, batch_paths


def main(cfg_path):
    assert osp.isfile(cfg_path), f"Config file {cfg_path} does not exist"
    cfg = load_config(cfg_path)
    assert osp.exists(cfg['input_img_dir']), f"Input image directory {cfg['input_img_dir']} does not exist"
    models = init_models(cfg)
    batch_frames, batch_paths = load_images(cfg['input_img_dir'])

    zipped = list(zip(batch_frames, batch_paths))
    for img_idx, (img, path) in enumerate(tqdm(zipped, total=len(zipped))):
        shapes = []
        for model_idx, model in enumerate(models):
            class_names = cfg['models'][model_idx]['class_names']
            results = model.detect(img)
            for obj in results:
                if class_names[obj.classID] not in cfg['models'][model_idx]['class_filter']:
                    continue
                shape_item = detbox_to_shape_rectangle(obj, class_names)
                shapes.append(shape_item)
        save_anylabeling_json(path, shapes, cfg['input_img_dir'], img.shape)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trt_multi_model_class_specific_prelabel.py config.yaml")
        exit(1)
    main(sys.argv[1])