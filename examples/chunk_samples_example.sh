

img_dir=/data/nofar/person_behavior/hbb_cls10_action_v0.5.9/images
out_dir=/home/lorenzo/test_chunk
chunk_size=1000

# 无法递归img_dir下的子目录，只能处理img_dir下的文件
python preprocessing/chunk_samples.py ${img_dir} ${out_dir} --chunk-size ${chunk_size}