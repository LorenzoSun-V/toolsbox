from dataset_utils import DetValidator
import os.path as osp

root_path = "/data/lorenzo/datasets/fire-smoke/fire/cls1_fire_v0.1.2"
pred_path = "/home/lorenzo/Code/Detection/yolov5/exps/fire/cls1_fire_v0.1.3/model5_b-1s_cuspretrain_20250427_cls1_fire_v0.1.3/val_cls1_fire_v0.1.2/labels"
gt_path = osp.join(root_path, "gts")
val_list_path = osp.join(root_path, "trainval", "val.txt")
names = {0: 'fire'}
# names = {0: 'smoke'}

detval = DetValidator(
    pred_path=pred_path,
    gt_path=gt_path,
    val_list_path=val_list_path,
    names=names,
)
detval.cal_metrics()