#! /bin/bash


input_model=/home/lorenzo/Code/Detection/ultralytics/exps/nofar/person_behavior/person_attribute_v0.2.9/model8_b32m_20250707_person_attribute_v0.2.9.onnx
outputmodel=/home/lorenzo/Code/Detection/ultralytics/exps/nofar/person_behavior/person_attribute_v0.2.9/model8_b32m_20250707_person_attribute_v0.2.9_1.onnx
numcls=10
keepTopK=300

python3 yolov8_add_postprocess.py --inputmodel ${input_model} --outputmodel ${outputmodel} --numcls ${numcls}
python3 yolov8_add_nms.py --model ${outputmodel} --numcls ${numcls} --keepTopK ${keepTopK}
