from dataset_utils import  Visualizer

# 1. 如何使用Visualizer
# img_dir = "/data/public/person_underground/images"  # 图像文件夹路径
# vis_dir = "./"  # 可视化结果保存路径
# visualizer = Visualizer(img_dir, vis_dir)
# yolo_label = "/data/public/person_underground/labels" # yolo格式的label文件夹路径
# visualizer.vis_yolo_style(yolo_label)

img_dir = "/data/public/coco/images/val2017"  # 图像文件夹路径
label_list_dir = "/data/public/coco/classes.txt"  # 图像文件夹路径
vis_dir = "./"  # 可视化结果保存路径
visualizer = Visualizer(img_dir, vis_dir)
# yolo_label = "/data/lorenzo/datasets/headshoulder/Headshoulder/Input/quarter_view/Annotations" # yolo格式的label文件夹路径
# visualizer.vis_yolo_style(yolo_label)
voc_label = "/data/public/coco/Annotations" # yolo格式的label文件夹路径
visualizer.vis_voc_style(voc_label, label_list_dir)