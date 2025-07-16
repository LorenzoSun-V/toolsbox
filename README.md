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


## 4. 数据预处理

<details>
<summary>1. 视频按时间戳切片</summary>

视频按时间戳切片脚本请参考[video_extractor_example.sh](./examples/video_extractor_example.sh)

使用FFmpeg代替OpenCV进行视频切片，减少灰色帧出现的情况。

```
video_dir=/data/video
out_dir=/data/video_clips
json_file=configs/video_extractor.json

# 无法递归video_dir下的子目录，只能处理video_dir下的文件
python preprocessing/video_extractor.py ${video_dir} ${out_dir} --json-file ${json_file} --log-level DEBUG
```

参数说明：
- `video_dir`：视频文件目录，脚本会处理该目录下的所有视频文件，但`不会递归`子目录。
- `out_dir`：输出目录，脚本会将提取的帧保存到该目录。
- `json_file`：配置文件，脚本会根据该文件中的配置进行视频切片。格式为 {"video.mp4": ["00:00:05"-"00:00:10", "00:01:00"-"00:01:05"]}，表示对video.mp4视频文件在5秒到10秒和1分钟到1分5秒的时间段进行切片。
- `log-level`：日志级别，默认为INFO。

</details>

<details>
<summary>2. 视频切帧</summary>

视频切帧脚本请参考[video_frame_extractor_example.sh](./examples/video_frame_extractor_example.sh)

使用FFmpeg代替OpenCV进行视频切帧，减少灰色帧出现的情况。

```
video_dir=/data/video
out_dir=/data/frame
skip=10

# 递归video_dir下的子目录，处理video_dir下的所有文件
python preprocessing/video_frame_extractor.py ${video_dir} ${out_dir} --skip ${skip} --log-level DEBUG
```

参数说明：
- `video_dir`：视频文件目录，脚本`会递归`video_dir下的子目录，处理video_dir下的所有文件
- `out_dir`：输出目录，脚本会将提取的帧保存到该目录。
- `json_file`：配置文件，脚本会根据该文件中的配置进行视频切片。格式为 {"video.mp4": ["00:00:05"-"00:00:10", "00:01:00"-"00:01:05"]}，表示对video.mp4视频文件在5秒到10秒和1分钟到1分5秒的时间段进行切片。
- `log-level`：日志级别，默认为INFO。

</details>

<details>
<summary>3. 根据不同类别拆分数据集样本</summary>

根据不同类别拆分数据集样本脚本请参考[dataset_splitter_example.sh](./examples/dataset_splitter_example.sh)

该脚本主要是为了将数据集中的样本按照类别进行拆分，方便后续的训练和验证。在前期标注时，可能会按照标注规则标注所有样本，但在训练时可能只需要某些类别的样本进行训练。因此该脚本会根据json文件中的配置，将包含`对应类别的样本`以及`负样本`复制到要拆分的数据集根目录下的`subdataset`目录下，并以数据集名称命名文件夹。数据集根目录必须包含`images`和`Annotations`文件夹，分别存放图片和VOC格式的标注文件。

```
data_dir=/data/dataset
config_file=configs/dataset_splitter.json

python preprocessing/dataset_splitter.py --source_dir ${data_dir} --config_file ${config_file}

# python preprocessing/dataset_splitter.py --config '{"fire-smoke": ["fire", "smoke"], "person": ["person"], "person_behavior": ["smoking", "tx"]}'
```

参数说明：
- `data_dir`：数据集根目录，必须包含`images`和`Annotations`文件夹，分别存放图片和VOC格式的标注文件。
- `config_file`：配置文件，脚本会根据该文件中的配置进行数据集拆分。格式为 `{"subdataset_name": ["class1", "class2", ...]}`，表示将包含`class1`, `class2`, ...类别的样本复制到`subdataset_name`文件夹下。脚本会在数据集根目录下创建`subdataset`文件夹，并在其中创建对应的子文件夹。

</details>

<details>
<summary>4. 数据集样本筛选</summary>

数据集样本筛选脚本请参考[filter_samples_by_json_example.sh](./examples/filter_samples_by_json_example.sh)

该脚本主要是为了将特定包含特定类别的样本筛选出来，符合要求的样本会连同标签一起复制到数据集根目录下的`select`目录下。

```
img_dir=/data/src
cls="legs tx"
num=1

# 无法递归img_dir下的子目录，只能处理img_dir下的文件
python preprocessing/filter_samples_by_json.py -s ${img_dir} -c ${cls} -n ${num}
```

参数说明：
- `img_dir`：图片文件目录，脚本会处理该目录下的所有图片文件，但`不会递归`子目录。
- `cls`：需要筛选的类别，多个类别用空格分隔。
- `num`：每个类别需要筛选的样本数量，默认为1，即当某个类别的数量超过num个就挑选出来。

</details>

<details>
<summary>5. 复制路径下的所有文件到指定路径</summary>

复制路径下的所有文件到指定路径脚本请参考[gather_samples_example.sh](./examples/gather_samples_example.sh)

该脚本主要是为了递归复制指定路径下的所有文件到目标路径（复制到同一个目录下），方便后续的处理。

```
src_dir=/data/src
dst_dir=/data/dst

# 可以递归src_dir下的子目录，处理所有文件，包括子目录下的文件，复制到dst_dir
python preprocessing/gather_samples.py -s ${src_dir} -d ${dst_dir}
```

参数说明：
- `src_dir`：源目录，脚本会递归处理该目录下的所有文件，包括子目录下的文件。
- `dst_dir`：目标目录，脚本会将处理后的文件复制到该目录。

</details>

<details>
<summary>6. 将大数据集标注分块</summary>

将大数据集标注分块脚本请参考[chunk_samples_example.sh](./examples/chunk_samples_example.sh)

该脚本主要是为了将X-AnyLabeling标注好的样本进行平均切块，方便持续迭代修改标签。

```
img_dir=/data/src
out_dir=/data/dst
chunk_size=1000

# 无法递归img_dir下的子目录，只能处理img_dir下的文件
python preprocessing/chunk_samples.py ${img_dir} ${out_dir} --chunk-size ${chunk_size}
```

参数说明：
- `img_dir`：图片文件目录，脚本会处理该目录下的所有图片文件，但`不会递归`子目录。如果图片没有对应的json标注文件，会跳过该图片。
- `out_dir`：输出目录，脚本会将处理后的文件复制到该目录。
- `chunk_size`：每个块的样本数量，默认为1000，即将样本平均切分为每块。

</details>