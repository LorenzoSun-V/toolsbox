from dataset_utils import DetValidator, OBBValidator, SegmentValidator
import os.path as osp


if __name__ == "__main__":
    # OBB 精度计算示例
    names = {
        0: "plane",
        1: "ship",
        2: "storage-tank",
        3: "baseball-diamond",
        4: "tennis-court",
        5: "basketball-court",
        6: "ground-track-field",
        7: "harbor",
        8: "bridge",
        9: "large-vehicle",
        10: "small-vehicle",
        11: "helicopter",
        12: "roundabout",
        13: "soccer-ball-field",
        14: "swimming-pool"
    }

    pred_paths = [
        "/home/lorenzo/Code/Detection/ultralytics/exps/public/DOTAv1/yolov8n-obb/val/labels",
        "/data/public/DOTAv1/val_output/yolov8n-obb_i8",
        "/data/public/DOTAv1/val_output/yolov8n-obb_u8",
        "/home/lorenzo/Code/Detection/ultralytics/exps/public/DOTAv1/yolov8s-obb/val/labels",
        "/data/public/DOTAv1/val_output/yolov8s-obb_i8",
        "/data/public/DOTAv1/val_output/yolov8s-obb_u8",
        "/home/lorenzo/Code/Detection/ultralytics/exps/public/DOTAv1/yolov8m-obb/val/labels",
        "/data/public/DOTAv1/val_output/yolov8m-obb_i8",
        "/data/public/DOTAv1/val_output/yolov8m-obb_u8",
        "/home/lorenzo/Code/Detection/ultralytics/exps/public/DOTAv1/yolov8l-obb/val/labels",
        "/data/public/DOTAv1/val_output/yolov8l-obb_i8",
        "/data/public/DOTAv1/val_output/yolov8l-obb_u8",
        "/home/lorenzo/Code/Detection/ultralytics/exps/public/DOTAv1/yolov8x-obb/val/labels",
        "/data/public/DOTAv1/val_output/yolov8x-obb_i8",
        "/data/public/DOTAv1/val_output/yolov8x-obb_u8",
    ]
    for pred_path in pred_paths:
        print("pred_path: ", pred_path)
        obbval = OBBValidator(
            pred_path=pred_path,
            gt_path="/data/public/DOTAv1/labels/val_original",
            val_list_path="/data/public/DOTAv1/val.txt",
            names=names)
        obbval.cal_metrics()
    # obbval = OBBValidator(
    #     pred_path="/data/public/DOTAv1/val_output/yolov8n-obb_u8",
    #     gt_path="/data/public/DOTAv1/labels/val_original",
    #     val_list_path="/data/public/DOTAv1/val.txt",
    #     names=names) 
    # obbval.cal_metrics()

    # HBB 精度计算示例
    # detval = DetValidator(
    #     pred_path="/lorenzo/bt_repo/ultralytics/runs/detect/val2/labels",
    #     gt_path="/data/bt/xray/cls13_xray-sub_v1.0/gts",
    #     val_list_path="/data/bt/xray/cls13_xray-sub_v1.0/trainval/val.txt",
    #     names={0:'embedding_gs', 1: 'fracture_gs', 2: 'fall_gs', 3: 'damage_gs', 4: 'rust_gs', 5: 'embedding_fl', 
    #            6: 'fracture_fl', 7: 'glue_fl', 8: 'scratch_fl', 9: 'fall_fl', 10: 'brokenlen_gs', 11: 'fracture_full_gs', 12: 'rust_full_gs'})
    # detval.cal_metrics()

    # InstanceSegment 精度计算示例
    # names = {}
    # for i in range(80):
    #     names[i] = str(i)
    # root_path = "/data/Datasets/public/coco_instanceseg"
    # root_path = "/data/Datasets/public/coco_instanceseg/mini-test"
    # segval = SegmentValidator(
    #     pred_path="/data/Datasets/public/coco_instanceseg/mini-test/images/output",
    #     gt_path=f"{root_path}/labels",
    #     val_list_path=f"{root_path}/val.txt",
    #     names=names)
    # segval.cal_metrics()