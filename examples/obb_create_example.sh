
root_dir=/path/to/your/data
img_dir=${root_dir}/images
voc_anno_list=${root_dir}/trainval
voc_label_list=${voc_anno_list}/classes.txt


python dataset_utils/split_dota.py \
    --data ${root_dir} 

python dataset_utils/dota2yolobb.py \
    ${voc_label_list} \
    --data ${root_dir}