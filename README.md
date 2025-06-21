<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2023-11-09 15:14:46
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2023-11-09 15:14:56
 * @Description:
-->
# 数据集相关

具体请参考[dataset_utils](./dataset_utils/README.md).

# 使用TensorRT多模型预标注

## 1. 配置文件

请参考[model_config.yaml](./inference/model_config.yaml)文件，配置需要使用的模型。

参数说明如下：
- `input_img_dir`：输入图片目录。
- `lib_paths`： 私有库文件路径。
- `models`：模型列表，每个模型包含以下参数：
    - `model_path`：引擎文件路径。
    - `class_names`：模型类别名称列表。
    - `class_filter`：模型类别过滤列表，会过滤掉不在列表中的类别。
    - `conf_threshold`：模型置信度阈值，低于阈值的框不标注。
    - `nms_threshold`：模型NMS阈值，会对框进行NMS。

## 2. 运行脚本

请参考[multi_model_inference_prelabel_trt.py](./multi_model_inference_prelabel_trt.py)文件，运行脚本。

```
python multi_model_inference_prelabel_trt.py inference/model_config.yaml
```