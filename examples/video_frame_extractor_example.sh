
video_dir=/data/nofar/material/liandongUgu/2025-07-09/facility_clip
out_dir=/home/lorenzo/test_frame
skip=25

# 递归video_dir下的子目录，处理video_dir下的所有文件
python preprocessing/video_frame_extractor.py ${video_dir} ${out_dir} --skip ${skip} --log-level DEBUG