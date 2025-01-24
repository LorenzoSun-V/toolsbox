###
 # @Author: 孙家辉 sunjiahui@boton-tech.com
 # @Date: 2023-11-08 07:44:36
 # @LastEditors: 孙家辉 sunjiahui@boton-tech.com
 # @LastEditTime: 2023-11-16 05:17:05
 # @Description: 
### 

root_dir=/data/bt/hw_multi/cls2_jg_v0.4_shangang4-2_20231116
img_dir=${root_dir}/images
voc_anno_dir=${root_dir}/voc_labels
voc_anno_list=${root_dir}/trainval
voc_label_list=${voc_anno_list}/label_list.txt

python dataset_utils/voc2yolo.py \
    SL MS \
    --xml-dir ${voc_anno_dir}

python dataset_utils/create_voc.py \
    --img_dir ${img_dir} \
    --voc_anno_dir  ${voc_anno_dir} \
    --voc_anno_list ${voc_anno_list} \
    --train_proportion 0.9

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
