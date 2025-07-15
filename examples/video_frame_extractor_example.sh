
root_path=/data/nofar/material/liandongUgu/2025-07-09/facility_clip
out_path=/home/lorenzo/test_frame
skip=25

# 递归video_dir下的子目录，处理video_dir下的所有文件
python preprocessing/video_frame_extractor.py ${root_path} ${out_path} --skip ${skip}