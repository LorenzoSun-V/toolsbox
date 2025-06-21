import sys
import yaml
import cv2
import json
import os
import os.path as osp
from inference.yolo import Yolov8Detector


# --------- Config Parsing --------- #
def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg



def main(cfg_path):
    assert osp.isfile(cfg_path), f"Config file {cfg_path} does not exist"
    cfg = load_config(cfg_path)
    model_path = cfg['models'][0]['model_path']
    lib_path = "/home/lorenzo/Code/toolsbox/libs/libyoloe2e-c.so"
    # model = Yolov8obbDetector(
    model = Yolov8Detector(
        library_path=lib_path,
        weights_path=model_path,
        conf_threshold=0.25,
        nms_threshold=0.45
    )
    for path in os.listdir("/data/lorenzo/datasets/temp/images"):
        print(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trt_multi_model_class_specific_prelabel.py config.yaml")
        exit(1)
    main(sys.argv[1])