<!--
 * @Author: 孙家辉 sunjiahui@boton-tech.com
 * @Date: 2023-11-09 00:33:09
 * @LastEditors: 孙家辉 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-23 06:07:51
 * @Description: 
-->
# 数据集Pipeline

## 目录

1. [HBB数据集Pipeline](#hbb数据集pipeline)
2. [OBB数据集Pipeline](#obb数据集pipeline)
3. [InstanceSegment数据集Pipeline](#instancesegment数据集pipeline)

### HBB数据集Pipeline

#### 1. 标注数据集

根据标注规范，使用labelImg工具标注，导出的数据集为voc格式，请确保数据集文件结构为：
```
dataset
├─ images
│    ├─ 1.jpg
│    ├─ 2.jpg
│    ├─ 3.jpg
│    ├─ ...
│    └─ 1000.jpg
└─ Annotations
    ├─ 1.xml
    ├─ 2.xml
    ├─ 3.xml
    ├─ ...
    └─ 1000.xml
```

#### 2. voc2yolo

使用[voc2yolo.py](./voc2yolo.py)将voc数据集转为yolo格式：
```
python dataset_utils/voc2yolo.py \
    --voc_label_list ${voc_label_list} \
    --xml-dir ${voc_anno_dir}
```
参数说明：
* voc_label_list: 数据集的类别txt文件，每行一个类别，顺序和定义保持一致。
* xml-dir: 存放voc标签的路径

eg. 数据集有`person`,`car`,`trunk`三类(标签id分别为0,1,2)，那么 voc_label_list 的文件应该为：
```
person
car
trunk
```

voc格式的标签地址为`/dataset/Annotations`，执行命令：

```
python dataset_utils/voc2yolo.py person car trunk --xml-dir /dataset/Annotations
```
最后生成txt格式的yolo标签，会默认保存在`/dataset/labels`，即voc标签同级目录下的`labels`文件夹，现在数据集文件结构为：
```
dataset
├─ images
│    ├─ 1.jpg
│    ├─ 2.jpg
│    ├─ 3.jpg
│    ├─ ...
│    └─ 1000.jpg
├─ labels
│    ├─ 1.txt
│    ├─ 2.txt
│    ├─ 3.txt
│    ├─ ...
│    └─ 1000.txt
└─ Annotations
    ├─ 1.xml
    ├─ 2.xml
    ├─ 3.xml
    ├─ ...
    └─ 1000.xml
```

#### 3. voc2coco

这一步操作会将voc数据集转为coco格式，同时会*划分yolo和coco两种格式的训练集和验证集*，划分会保证两种格式的训练集和验证集图片保持一致。</br>
1. 使用[create_voc.py](./create_voc.py)划分训练集和验证集：
    ```
    python dataset_utils/create_voc.py \
        --img_dir ${img_dir} \
        --voc_anno_dir  ${voc_anno_dir} \
        --voc_anno_list ${voc_anno_list} \
        --train_proportion 0.9
    ```
    参数说明：
    * --img_dir: 图像路径，用于划分yolo格式的训练集和验证集时，在图像名前加上绝对路径
    * --voc_anno_dir: 存放voc标签的路径，和步骤2中的`--xml-dir`相同
    * --voc_ann_list: 存放划分训练集和验证集文件的路径
    * --train_proportion: 训练集占图像的比例

    假设`--voc_anno_list /dataset/trainval`，该步骤完成，数据集文件结构为（在`/dataset/trainval`文件夹中新增了训练集和验证集的划分）：
    ```
    dataset
    ├─ images
    │    ├─ 1.jpg
    │    ├─ 2.jpg
    │    ├─ 3.jpg
    │    ├─ ...
    │    └─ 1000.jpg
    ├─ labels
    │    ├─ 1.txt
    │    ├─ 2.txt
    │    ├─ 3.txt
    │    ├─ ...
    │    └─ 1000.txt
    ├─ trainval
    │    ├─ train.txt
    │    ├─ train_stem_.txt
    │    ├─ val.txt
    │    └─ val_stem.txt
    └─ Annotations
        ├─ 1.xml
        ├─ 2.xml
        ├─ 3.xml
        ├─ ...
        └─ 1000.xml
    ```
    `train.txt`和`val.txt`是yolo格式的训练集和验证集划分，文件中是图片的绝对路径；`train_stem.txt`和`val_stem.txt`只有不包含路径和文件后缀的文件名，在后面的步骤会用到。

2. 创建数据集的类别文件（每行保存一个类别名称，以txt文件保存）后，使用[x2coco.py](./x2coco.py)将voc转为coco：
    ```
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
    ```
    参数说明：
    * --voc_anno_dir: 存放voc标签的路径
    * --voc_anno_list: 仅包含文件名(不含文件后缀)的txt文件
    * --voc_label_list: 数据集的类别文件，每行保存一个类别名称，以txt文件保存
    * --output_dir: 存放划分训练集和验证集文件的路径
    * --voc_out_name

    假设`--output_dir /dataset/trainval`，`--voc_label_list /dataset/label_list.txt`，该步骤完成，数据集文件结构为（在`/dataset/trainval`文件夹中新增了coco格式的训练集和验证集）：
    ```
    dataset
    ├─ images
    │    ├─ 1.jpg
    │    ├─ 2.jpg
    │    ├─ 3.jpg
    │    ├─ ...
    │    └─ 1000.jpg
    ├─ labels
    │    ├─ 1.txt
    │    ├─ 2.txt
    │    ├─ 3.txt
    │    ├─ ...
    │    └─ 1000.txt
    ├─ trainval
    │    ├─ label_list.txt
    │    ├─ train.json
    │    ├─ train.txt
    │    ├─ train_stem_.txt
    │    ├─ val.json
    │    ├─ val.txt
    │    └─ val_stem.txt
    └─ Annotations
        ├─ 1.xml
        ├─ 2.xml
        ├─ 3.xml
        ├─ ...
        └─ 1000.xml
    ```

yolo格式的训练集和验证集会以`train.txt`和`val.txt`保存，coco格式的训练集和验证集会以`train.json`和`val.json`保存。
以上步骤脚本，可以参考[create_voc2coco.sh](../create_voc2coco.sh)

#### 4. visualize
如果按照以上步骤生成了数据集，可以使用`Visualizer`可视化数据集。具体教程可参考[examples](../dataset_utils_example.ipynb)。


#### 5. calculate metrics
首先运行[voc2xyxy.py](./voc2xyxy.py)获得当前验证集的Ground Truths：
```
python dataset_utils/voc2xyxy.py \
    --voc_label_list ${voc_label_list} \
    --xml-dir ${voc_anno_dir} \
    --val-list ${voc_ann_list}/val_stem.txt
```
参数说明：
* voc_label_list: 数据集的类别txt文件，每行一个类别，顺序和定义保持一致。
* xml-dir: 存放voc标签的路径
* val-list: 仅包含验证集图片名(不含文件后缀)的txt文件，即步骤3中生成的`val_stem.txt`

该步骤完成，会生成`gts`文件夹保存验证集的Ground Truths(cls,x1,y1,x2,y2)，数据集文件结构为：

```
dataset
├─ gts
│    ├─ 5.txt
│    ├─ 8.txt
│    ├─ 10.txt
│    ├─ ...
│    └─ 1000.txt
├─ images
│    ├─ 1.jpg
│    ├─ 2.jpg
│    ├─ 3.jpg
│    ├─ ...
│    └─ 1000.jpg
├─ labels
│    ├─ 1.txt
│    ├─ 2.txt
│    ├─ 3.txt
│    ├─ ...
│    └─ 1000.txt
├─ trainval
│    ├─ label_list.txt
│    ├─ train.json
│    ├─ train.txt
│    ├─ train_stem_.txt
│    ├─ val.json
│    ├─ val.txt
│    └─ val_stem.txt
└─ Annotations
    ├─ 1.xml
    ├─ 2.xml
    ├─ 3.xml
    ├─ ...
    └─ 1000.xml
```

使用`DetValidator`计算验证集精度：
```
from dataset_utils import DetValidator
detval = DetValidator(
    pred_path="/lorenzo/bt_repo/ultralytics/runs/hwir/cls2_20231107_1floor_val/labels",  # 预测结果文件夹路径，默认conf_thresh=0.45,nms_thresh=0.65
    gt_path="/data/bt/hw_multi/raw/ir/1floor/20231020_1floor/gts",  # 真值文件夹路径
    val_list_path="/data/bt/hw_multi/raw/ir/1floor/20231020_1floor/trainval/v0.1/val.txt",  # 预测的图像文件名列表文件路径
    names={0:'SL', 1:'MS'}  # 类别字典
)
detval.cal_metrics()
```

具体可参考[examples](../dataset_utils_example.ipynb)。


### OBB数据集Pipeline

#### 1. 标注数据集

用标注工具标注OBB标注框，并导出为[DOTA格式](https://docs.ultralytics.com/datasets/obb/dota-v2/)

标注完后，目录结构如下：

```
dataset
├─ images
│    ├─ 1.jpg
│    ├─ 2.jpg
│    ├─ 3.jpg
│    ├─ ...
│    └─ 1000.jpg
└─ labelTxt
    ├─ 1.txt
    ├─ 2.txt
    ├─ 3.txt
    ├─ ...
    └─ 1000.txt
```

#### 2. 划分训练、验证和测试集

使用[split_dota.py](./split_dota.py)划分训练集、验证集和测试集：

```
python dataset_utils/split_dota.py \
    --data ${root_dir} \
    --ratio 0.8 0.2 0
```

参数说明：
* --data: 数据集根目录
* --ratio: 训练集、验证集和测试集的比例

执行完后，目录结构为，train_original和val_original分别存放着DOTA格式的标签：

```
dataset
├── images
│   ├── train
│   └── val
└── labels
    ├── train_original
    └── val_original
```

#### 3. DOTA2YOLOBB

使用[dota2yolobb.py](./dota2yolobb.py)将DOTA格式标签转为YOLO-OBB格式：

```
python dataset_utils/dota2yolobb.py \
    ${label_list} \
    --data ${root_dir}
```

参数说明：
* label_list: 数据集类别txt文件，每行一个类别，顺序和定义保持一致。
* --data: 数据集根目录

执行完后，目录结构为，train和val分别存放着YOLO-OBB格式的标签：

```
dataset
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    ├── train_original
    ├── val
    └── val_original
```

#### 4. calculate metrics

使用`OBBValidator`计算验证集精度：
```
from dataset_utils import OBBValidator
obbval = OBBValidator(
        pred_path="/lorenzo/bt_repo/ultralytics/runs/hw_obb/model8_b32s-obb_20241114_cls2_hw_obb_v0.1/detect/labels/output",
        gt_path="/data/bt/hw_obb/20241114/labels/val_original",
        val_list_path="/data/bt/hw_obb/20241114/val.txt",
        names={0:'SL', 1:'MS'}) 
obbval.cal_metrics()
```

其中：
* pred_path是推理时每张图片对应保存的预测标签，格式为 `[x_center, y_center, w, h, radian, conf, cls]`。
如果直接使用ultralytic库保存验证结果，需更改*ultralytics/engine/results.py*中*Results*类中的*save_txt*函数：
    ```
    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
                - For detections: `class confidence x_center y_center width height`
                - For classifications: `confidence class_name`
                - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                #! original ultralytics code
                # c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                # line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                # if masks:
                #     seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                #     line = (c, *seg)
                # if kpts is not None:
                #     kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                #     line += (*kpt.reshape(-1).tolist(),)
                # line += (conf,) * save_conf + (() if id is None else (id,))
                # texts.append(("%g " * len(line)).rstrip() % line)

                #! new code, save val label format: x1, y1, x2, y2, conf, cls, id, x1, y1, x2, y2, conf, cls, id, ...
                c = int(d.cls)
                conf = float(d.conf)
                id = None if d.id is None else int(d.id.item())
                # Convert to xyxy format
                if is_obb:
                    # coords = d.xyxyxyxyn.view(-1)
                    coords = d.xywhr.view(-1)
                else:
                    coords = d.xyxy.view(-1)  # Use xyxy format directly
                # Construct line in the format: xyxy, conf, class
                if save_conf:
                    line = (*coords, conf, c)
                else:
                    line = (*coords, c)
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (*seg, conf, c)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                if id is not None:
                    line += (id,)

                # Convert line to text format and append to texts
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)
    ```
* gt_path是标注数据集时，导出的DOTA格式标签，格式为 `[x1, y1, x2, y2, x3, y3, x4, y4, cls_name]`，验证集gts在`labels/val_original`路径下。
* val_list_path是验证集的图片路径，可以用shell脚本 `find "/data/images/val" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) > "/data/val.txt"`生成。
* names是类别映射。

### InstanceSegment数据集pipeline

#### 1. 标注数据集

用标注工具标注多边形框，并导出为YOLO实例分割格式。

标注完后，目录结构如下：
```
dataset
├─ images
│    ├─ 1.jpg
│    ├─ 2.jpg
│    ├─ 3.jpg
│    ├─ ...
│    └─ 1000.jpg
└─ labels
    ├─ 1.txt
    ├─ 2.txt
    ├─ 3.txt
    ├─ ...
    └─ 1000.txt
```

#### 2. 划分训练、验证和测试集

可以参考[split.py](./split.py)划分训练集、验证集和测试集。

#### 3. 精度计算

先使用bt_alg_api项目中yoloseg的test-precious计算并保存每张验证集图片的预测结果，每张图片都会有一个同名`txt`文件保存bbox信息，以及一个同名`bin`文件保存mask信息。

参考代码：
```
names = {}
for i in range(80):
    names[i] = str(i)
root_path = "/data/Datasets/public/coco_instanceseg"
root_path = "/data/Datasets/public/coco_instanceseg/mini-test"
segval = SegmentValidator(
    pred_path="/data/Datasets/public/coco_instanceseg/mini-test/images/output",
    gt_path=f"{root_path}/labels",
    val_list_path=f"{root_path}/val.txt",
    names=names)
segval.cal_metrics()
```

* pred_path: test-precious保存的预测结果。
* gt_path: 标注数据集时，导出的YOLO格式标签。
* val_list_path: 验证集图片路径，一定要是绝对路径。
* names: 类别映射。