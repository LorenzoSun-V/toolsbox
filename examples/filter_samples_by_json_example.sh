
img_dir=/data/nofar/material/liandongUgu/2025-07-09/person_behavior_labeling/merged_0711_0714
cls="legs tx"

# 无法递归img_dir下的子目录，只能处理img_dir下的文件
python preprocessing/filter_samples_by_json.py -s ${img_dir} -c ${cls}