## 前置条件

* 推荐 Python==3.8
* gcc >= 5 (at least C++14 support)

## 安装

### 从源码安装

* 运行以下命令以安装最新版本

```shell
# 使用 Anaconda【推荐】
git clone https://github.com/ProjectVaquitai/Vaquitai.git
cd Vaquitai
conda create --name vaquitai python==3.8
conda activate vaquitai
pip install -v -e .[all]
```

```shell
# 不使用 Anaconda
cd Vaquitai
pip install -v -e .[all]
```

## 快速上手

### 数据处理

```shell
# 本地启动
streamlit run app.py
```

```shell
# ssh 远程终端中启动
streamlit run app.py --server.address 0.0.0.0 --server.port 80

```

### 构建配置文件

* 配置文件包含一系列全局参数和用于数据处理的算子列表。您需要设置:
  * 全局参数：输入/输出 数据集路径，worker 进程数量等。
  * 算子列表：列出用于处理数据集的算子及其参数。
* 您可以通过如下方式构建自己的配置文件:
  * 修改我们的样例配置文件 [`vaquitai.yaml`](configs/demo/vaquitai.yaml)。该文件包含了**所有**算子以及算子对应的默认参数。您只需要**移除**不需要的算子并重新设置部分算子的参数即可。

* 基础的配置项格式及定义如下图所示

  ![基础配置项格式及定义样例](https://img.alicdn.com/imgextra/i4/O1CN01xPtU0t1YOwsZyuqCx_!!6000000003050-0-tps-1692-879.jpg "基础配置文件样例")

### 输入数据格式
以下参数均为非必选
- 若想处理图像数据，请带有 `image`
- 若想处理文字数据，请带有 `text`

```json
{
"image": "./demos/vaquitai/data/cifar10_cl/cat_s_000081.png", 
"text": "Today is Sunday and it's a happy day!",
"annotations": ["cat"]
}
```

### 运行
首次运行会进行模型下载

![数据处理执行流程](https://datacentric-1316957999.cos.ap-beijing.myqcloud.com/data-centric/app_image/home/process.jpg)

### 算子支持列表
| 类型 |      OP      |             名称             |             参数示例             |                            参数说明                             |                          描述                          |
|:----:|:------------:|:---------------------------:|:--------------------------------:|:--------------------------------------------------------------:|:------------------------------------------------------:|
|  减  | Deduplicator |       image_deduplicator     |          method: phash           | hash method for image. One of [phash, dhash, whash, ahash] |   运用哈希判断数据集中是否有重复图片   |
|      |  Mycleanlab  |     cleanvision_mycleanlab   | issues: ["is_low_information_issue"] | Please select the desired field to be cleaned from the list. ["is_odd_size_issue", "is_odd_aspect_ratio_issue", "is_low_information_issue", "is_light_issue", "is_grayscale_issue", "is_dark_issue", "is_blurry_issue", "is_exact_duplicates_issue", "is_near_duplicates_issue"] | 进行图像等级的各方面筛查，您可选择列表中的一个或多个所需属性进行检查 |
|  看  |  Generator   |     feature_reduce_generator  |               null               |                               null                               |                    特征降维                    |
|      |  Generator   | image_feature_extract_generator |               null               |                               null                               |                    特征提取                    |
|      |  Generator   |     image_caption_generator   |               null               |                               null                               |                    图像描述                    |
