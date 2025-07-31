
task=hbb
pred_path=/home/lorenzo/Code/Detection/ultralytics/exps/person_behavior/hbb_action_cls10_v0.6.1/model8_b128m_20250726_hbb_action_cls10_v0.6.1/val/labels
gt_path=/data/nofar/person_behavior/hbb_action_cls10_v0.6.1/gts
val_list_path=/data/nofar/person_behavior/hbb_action_cls10_v0.6.1/trainval/val.txt
names_file=/data/nofar/person_behavior/cls10.txt

python calculate_metric.py ${task} \
    --pred_path ${pred_path} \
    --gt_path ${gt_path} \
    --val_list_path ${val_list_path} \
    --names_file ${names_file} \