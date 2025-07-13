
root_dir=/path/to/your/data
img_dir=${root_dir}/images
voc_anno_dir=${root_dir}/Annotations
voc_anno_list=${root_dir}/trainval
voc_label_list=${voc_anno_list}/classes.txt


python dataset_utils/voc2yolo.py \
    --voc-label-list ${voc_label_list} \
    --xml-dir ${voc_anno_dir}

python dataset_utils/create_voc.py \
    --img_dir ${img_dir} \
    --voc_anno_dir  ${voc_anno_dir} \
    --voc_anno_list ${voc_anno_list} \
    --train_proportion 0.8

python dataset_utils/x2coco.py \
    --dataset_type voc \
    --voc_anno_dir ${voc_anno_dir} \
    --voc_anno_list ${voc_anno_list}/train_stem.txt \
    --voc_label_list ${voc_label_list} \
    --output_dir ${voc_anno_list} \
    --voc_out_name train.json

python dataset_utils/x2coco.py \
    --dataset_type voc \
    --voc_anno_dir ${voc_anno_dir} \
    --voc_anno_list ${voc_anno_list}/val_stem.txt \
    --voc_label_list ${voc_label_list} \
    --output_dir ${voc_anno_list} \
    --voc_out_name val.json

python dataset_utils/voc2xyxy.py \
    --voc-label-list ${voc_label_list} \
    --val-list ${voc_anno_list}/val_stem.txt \
    --xml-dir ${voc_anno_dir}