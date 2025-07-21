#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-10-17 15:15:04
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2025-02-26 10:29:43
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/v2/python/yolov5/run_yolo_nms.sh
### 

input_model=/home/sysadmin/jack/iscyy/yoloair/runs/train/yolov5s_cbam_cls13_xray-sub_v1.02/weights/best.onnx
outputmodel=/home/sysadmin/jack/iscyy/yoloair/runs/train/yolov5s_cbam_cls13_xray-sub_v1.02/weights/model5_cbam_b64s_20241112_cls13_xray-sub_v1.0_b1_1.onnx
numcls=13
keepTopK=100

python add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
