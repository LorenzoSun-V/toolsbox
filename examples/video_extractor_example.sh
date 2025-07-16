
video_dir=/data/nofar/material/liandongUgu/2025-07-09/facility
out_dir=/home/lorenzo/test_clip
json_file=configs/video_extractor.json

# 无法递归video_dir下的子目录，只能处理video_dir下的文件
python preprocessing/video_extractor.py ${video_dir} ${out_dir} --json-file ${json_file} --log-level DEBUG