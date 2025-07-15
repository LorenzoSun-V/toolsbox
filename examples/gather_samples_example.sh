
src_dir=/data/nofar/material/liandongUgu/2025-07-09/person_behavior_labeling/merged_0711_0714
dst_dir=/home/lorenzo/test_gather

# 可以递归src_dir下的子目录，处理所有文件，包括子目录下的文件，复制到dst_dir
python preprocessing/gather_samples.py -s ${src_dir} -d ${dst_dir}