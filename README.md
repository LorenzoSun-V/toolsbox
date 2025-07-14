<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2023-11-09 15:14:46
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2023-11-09 15:14:56
 * @Description:
-->
# 数据集相关

## 1. 数据集制作

具体说明请参考[dataset_utils](./dataset_utils/README.md).

HBB脚本请参考[hbb_create_example.sh](./examples/hbb_create_example.sh)

OBB脚本请参考[obb_create_example.sh](./examples/obb_create_example.sh)


## 2. 数据集合并

<details>
<summary>HBB数据集合并</summary>

HBB脚本请参考[hbb_merge_example.sh](./examples/hbb_merge_example.sh)

HBB数据集合并，以下示例脚本会将`/src/path/to/dataset1`, `/src/path/to/dataset2`, `/src/path/to/dataset3`合并至`/dst/path/to/merged/dataset`，并保持合并后的训练集和验证集划分一致：
```
python preprocessing/merge_hbb_dataset.py
    /src/path/to/dataset1 /src/path/to/dataset2 /src/path/to/dataset3
    /dst/path/to/merged/dataset
```

参数说明：
- `/src/path/to/dataset1`, `/src/path/to/dataset2`, ...: 需要合并的数据集路径，每个数据集需要根据[hbb_create_example.sh](./examples/hbb_create_example.sh)制作数据集，获得必要的文件。
- `/dst/path/to/merged/dataset`: 合并完后，数据集的目标路径。

</details>

<details>
<summary>OBB数据集合并</summary>

OBB脚本请参考[obb_merge_example.sh](./examples/obb_merge_example.sh)

OBB数据集合并，以下示例脚本会将`/src/path/to/dataset1`, `/src/path/to/dataset2`, `/src/path/to/dataset3`合并至`/dst/path/to/merged/dataset`，并保持合并后的训练集和验证集划分一致：
```
python preprocessing/merge_obb_dataset.py
    /src/path/to/dataset1 /src/path/to/dataset2 /src/path/to/dataset3
    /dst/path/to/merged/dataset
```

参数说明：
- `/src/path/to/dataset1`, `/src/path/to/dataset2`, ...: 需要合并的数据集路径，每个数据集需要根据[obb_create_example.sh](./examples/obb_create_example.sh)制作数据集，获得必要的文件。
- `/dst/path/to/merged/dataset`: 合并完后，数据集的目标路径。
</details>


## 3. 使用TensorRT多模型预标注

1. 配置文件

    请参考[model_config.yaml](./inference/model_config.yaml)文件，配置需要使用的模型。

    参数说明如下：
    - `input_img_dir`：输入图片目录。
    - `models`：模型列表，每个模型包含以下参数：
        - `model_path`：引擎文件路径。
        - `model_type`：模型类型，例如 yoloe2e, yolov8等。
        - `class_names`：模型类别名称列表。
        - `class_filter`：模型类别过滤列表，会过滤掉不在列表中的类别。
        - `conf_threshold`：模型置信度阈值，低于阈值的框不标注。
        - `nms_threshold`：模型NMS阈值，会对框进行NMS。

2. 运行脚本

    hbb预标注:

    请参考[multi_hbb_inference_prelabel_trt.py](./multi_hbb_inference_prelabel_trt.py)文件，运行脚本。
    ```
    python multi_hbb_inference_prelabel_trt.py inference/model_config.yaml
    ```