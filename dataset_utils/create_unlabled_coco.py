'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-12-19 05:21:40
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2023-12-19 06:07:50
Description: Create unlabeled datasets in coco format.
'''
import cv2
import argparse
import os.path as osp
import json
from glob import glob
from tqdm import tqdm


def create_unlabeled_coco(train_dir, image_dir, out_dir):
    anns_train = json.load(open(train_dir, 'r'))
    train_id_max = anns_train['images'][-1]['id']

    image_dirs = glob(osp.join(image_dir, "*.[jJ][pP][gG]*")) + glob(osp.join(image_dir, "*.[jJ][pP][eE][gG]*")) + glob(osp.join(image_dir, "*.[pP][nN][gG]*"))
    image_info = []
    for index, image_dir in tqdm(enumerate(image_dirs)):
        image_name = osp.basename(image_dir)
        h, w, _ = cv2.imread(image_dir).shape
        image_id = train_id_max + index + 1
        image_info.append(
            {"file_name": image_name, 
             "height": round(float(h), 1), 
             "width": round(float(w), 1), 
             "id": image_id}
        )

    unlabeled_json = {
    'images': image_info,
    'annotations': [],
    'categories': anns_train['categories'],
    }

    with open(out_dir, 'w') as f:
        json.dump(unlabeled_json, f)


def main():
    parser = argparse.ArgumentParser(description="create unlabeled datasets")
    parser.add_argument('output_dir', type=str, help='path of output json')
    parser.add_argument('train_dir', type=str, help='path of train json')
    parser.add_argument('image_dir', type=str, help='path of unlabeled images')
    args = parser.parse_args()

    assert args.output_dir.lower().endswith('.json'), "Error: The output file must have a .json extension."
    assert osp.exists(args.train_dir), f"train-dir '{args.train_dir}' is not existed! Please check!"
    assert osp.exists(args.image_dir), f"image-dir '{args.image_dir}' is not existed! Please check!"
    create_unlabeled_coco(args.train_dir, args.image_dir, args.output_dir)


if __name__ == '__main__':
    main()
