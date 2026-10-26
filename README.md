# ObjectDetection

一个**独立、通用的目标检测工程工具包**，基于 **Ultralytics YOLO**。仓库根目录本身就是一个 Python 包（`ObjectDetection`）：把**它的父目录**加入 `sys.path`，即可用 `from ObjectDetection import ...` 按需调用，不依赖任何特定项目。覆盖从**视频抽帧 → 标注格式转换 → 标签治理 → 数据集划分 → 训练 → 验证 → 推理 / 高 DPI 出图 → 模型导出 → 复杂度统计 → 训练指标对比 → 视觉大模型报告**的完整链路。

---

## 环境与依赖

Python ≥ 3.10（`preprocess.py` 使用 `set | None` 注解，3.9 导入即抛 `TypeError`；开发环境为 3.12 / 3.13）。

```bash
pip install ultralytics opencv-python numpy pillow tqdm lxml matplotlib pandas requests
```

| 依赖            | 用途                                               | 涉及模块                                                                             |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `ultralytics`   | YOLO / YOLOWorld 训练、验证、推理、导出、COCO 转换 | `train` `val` `predict` `high_dpi_predict` `export` `convert` `calc_complexity_yolo` |
| `opencv-python` | 图像读写、画框、显示、视频解码                     | `predict` `high_dpi_predict` `video`                                                 |
| `numpy`         | 坐标数组运算、指标统计                             | `predict` `high_dpi_predict` `convert` `compare_metrics`                             |
| `pillow`        | 中文文本绘制、ICC 配置处理、图片收集               | `predict` `high_dpi_predict` `preprocess`                                            |
| `tqdm`          | 进度条                                             | `convert` `preprocess` `dataset` `systools` `video`（ultralytics 内部也用）          |
| `lxml`          | VOC XML 解析                                       | `convert`                                                                            |
| `matplotlib`    | 训练曲线绘制、中文字体与论文白底样式               | `compare_metrics`                                                                    |
| `pandas`        | `results.csv` 读取与统计计算                       | `compare_metrics`                                                                    |
| `requests`      | 调用 OpenAI 兼容的视觉大模型接口                   | `open_api`（**惰性导入**，不装也不影响其它模块）                                     |

---

## 目录结构

```
ObjectDetection/                 # 仓库根目录即 Python 包
├── __init__.py                  # 统一 re-export + __all__（49 个符号，不含 high_dpi_predict）
├── convert.py                   # 标注格式转换：COCO / 百度 COCO / VOC → YOLO txt
├── preprocess.py                # 图像与标签预处理、标签类别治理、类别计数
├── dataset.py                   # 图标签配对检查、数据集划分、分块收集
├── train.py                     # YOLO 训练封装
├── val.py                       # 模型验证
├── predict.py                   # 推理（内置绘图 / 自绘中文标签两套）
├── high_dpi_predict.py          # 高 DPI 结果图（Times New Roman 标签 / DPI 元数据 / 放大输出）
├── export.py                    # 模型导出（ONNX / TensorRT / OpenVINO …）
├── model_utils.py               # 训练产物定位（best.pt / last.pt）
├── calc_complexity_yolo.py      # 模型复杂度统计（参数量 / GFLOPs）
├── compare_metrics.py           # 训练指标对比可视化（results.csv → 曲线 / 统计）
├── systools.py                  # 通用文件操作（建目录、复制、删除、批量重命名）
├── video.py                     # 视频抽帧（单文件 / 批量 / 递归）
├── open_api.py                  # 视觉大模型接口调用（OpenAI 兼容 / 流式 / 多模态）
├── README.md
└── update.bat                   # Windows 一键提交并推送脚本
```

---

## 功能总览

### 🎬 数据获取 — `video.py`

| 函数             | 作用                                               |
| ---------------- | -------------------------------------------------- |
| `extract_frames` | 主入口：批量抽帧，含配置打印、逐视频报告与汇总统计 |
| `process_video`  | 抽取单个视频的帧                                   |
| `collect_videos` | 从文件或文件夹收集视频                             |

### 🔄 标注转换 — `convert.py`

| 函数             | 作用                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `coco_to_txt`    | 标准 COCO json → YOLO txt（官方转换器，按 `cls91to80` 映射类别）      |
| `coco_txt_BaiDu` | 百度风格 COCO json → YOLO txt（自解析，不做类别映射，跳过 `iscrowd`） |
| `voc_to_txt`     | VOC XML → YOLO txt（需提供类别名→ID 映射字典）                        |

### 🧹 标签治理 — `preprocess.py`

| 函数                         | 作用                                                                       |
| ---------------------------- | -------------------------------------------------------------------------- |
| `count_labels`               | 统计各类别标注数量，支持 YOLO txt + VOC xml，可配 `names` 显示中文类名     |
| `replace_labels_with_target` | 把标签文件里**指定类别**的 ID 改成目标类别（`origin_id="all"` 为全量替换） |
| `batch_replace_labels`       | 上者的文件夹批量版，返回改动统计                                           |
| `fill_empty_labels`          | 为缺失标签的图片生成空标签文件（YOLO 背景图），支持 `dry_run` 预览         |
| `normalize_labels`           | 把标签坐标夹紧到 1.0 以内                                                  |
| `rm_icc_profile`             | 删除 PNG 的 ICC 色彩配置（部分标注工具对带 ICC 的图兼容性差）              |
| `collect_images`             | 收集目录下所有图片                                                         |
| `IMAGE_EXTS`                 | 模块级常见图片扩展名集合                                                   |

### 📂 数据集组织 — `dataset.py`

| 函数                           | 作用                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| `check_and_copy_missing_files` | 只查一个方向：图片多则找缺标签的图，标签多则找缺图的标签；`meta_path` 再比对一份元数据图片 |
| `spilt_train_test`             | 划分 train/val（可选 test）并复制落盘，支持负样本混入与固定随机种子                        |
| `split_meta`                   | 把目录按 `split_num` 分成若干份（分布式标注 / 分批处理用）；图标签数量不一致时静默返回     |
| `collect_meta`                 | 收集 `split_meta` 各子目录的文件；`save=True` 另复制一份到 `./meta`                        |
| `copy_files`                   | 复制文件，同名跳过；`max_num` 随机抽样，默认交互确认（`assume_yes=True` 跳过）             |
| `rm_files`                     | 删除指定文件**及其同名标签**（保证配对不被破坏）                                           |

### 🏋️ 训练 / 验证 / 推理 / 导出

| 模块                      | 函数                   | 作用                                                                                                                                                              |
| ------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.py`                | `train`                | YOLO 训练封装：`epochs` / `batch` / `imgsz` / `optimizer` / `lr` / `patience` / `mosaic` / `close_mosaic` / `seed_change`（False 固定 `seed=1`）/ `yolo_world` 等 |
| `val.py`                  | `evaluate`             | 验证：`iou`（NMS 阈值）/ `batch` / `rect` / `imgsz` / `conf`（默认 0.001），返回 ultralytics 验证结果；`rect` 口径见[注意点 9](#9-验证的-rect-口径不可混用)       |
| `predict.py`              | `predict`              | 用 ultralytics 内置绘图推理：`conf` / `iou` / `save` / `show` / `stream` / `verbose` / `yolo_world`                                                               |
| `predict.py`              | `advanced_predict`     | **自绘检测框**，支持中文字体与按 ID 覆盖类别名（未给出的沿用模型名称）；额外支持 `iou` / `yolo_world`                                                             |
| `predict.py`              | `add_pillow_text`      | 在图像数组上写文本（走 Pillow，解决 OpenCV 不支持中文）                                                                                                           |
| `predict.py`              | `get_infer_data`       | `typing=InferDataType.IMAGE`（随机抽样，`max_num` 上限）/ `InferDataType.DIR`（整目录）；`recursive` 控制子目录；扩展名规则见注意点 11                            |
| `export.py`               | `export_model`         | 导出 ONNX / TensorRT / OpenVINO / TorchScript；`yolo_world` 可选                                                                                                  |
| `model_utils.py`          | `get_best_last_model`  | 定位 `best.pt` / `last.pt`；`index` 指定 `./runs/detect/train{index}/weights`                                                                                     |
| `calc_complexity_yolo.py` | `calc_complexity_yolo` | 统计参数量（M）与计算量（GFLOPs），`fuse` 控制是否 BN 折叠，`imgsz` 参与 GFLOPs 计算                                                                              |

### 🖼️ 高 DPI 结果图 — `high_dpi_predict.py`

| 函数                              | 作用                                                                        |
| --------------------------------- | --------------------------------------------------------------------------- |
| `predict_high_dpi`                | 主入口：批量推理 → 画框与标签 → 按 `dpi` / `scale` 输出，返回已保存路径列表 |
| `draw_detections`                 | 在 BGR 画布上画框（OpenCV）+ 标签（Pillow）；标签自动避让边界               |
| `save_with_dpi`                   | 按指定 DPI 保存 PNG / JPEG（JPEG quality=95）                               |
| `collect_images`                  | 单图 / 目录（递归）/ 列表 → 图片路径列表                                    |
| `resolve_font_path` / `load_font` | 字重名或字体路径 → 字体对象；字体缺失时退回 Pillow 默认字体                 |

> 未并入 `__init__.py`，需从子模块导入：`from ObjectDetection.high_dpi_predict import predict_high_dpi`；导入条件见[注意点 13](#13-high_dpi_predict-使用前的注意事项)。

### 🤖 视觉大模型接口 — `open_api.py`

| 函数                    | 作用                                                     |
| ----------------------- | -------------------------------------------------------- |
| `OpenAPI`               | OpenAI 兼容接口的流式对话客户端，支持**图文多模态**输入  |
| `chat_stream`           | 函数式入口：一次调用拿到完整回复文本（后端生成报告常用） |
| `getSystemPrompt`       | 读取 system prompt 文本文件                              |
| `DEFAULT_SYSTEM_PROMPT` | 未指定 system prompt 时的默认文案                        |

### 📈 训练指标对比 — `compare_metrics.py`

| 函数                 | 作用                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `compare_metrics`    | **总入口**：读一个或多个 `results.csv`，出对比曲线 + 末 N epoch 统计                      |
| `plot_curves`        | 唯一出图函数（曲线叠加，可选末 N 均值参考线）                                             |
| `list_metrics`       | 罗列可选指标（编号 / key / 列名 / 趋势），打印编号表并返回 DataFrame                      |
| `resolve_metrics`    | 把编号 / key / 完整列名 / 模糊子串 / `Metric` 解析成 `Metric` 列表                        |
| `expand_csv_inputs`  | str / Path / 目录 / 列表 → `results.csv` 路径列表（去重、保序；`recursive` 控制目录递归） |
| `load_results_csv`   | 读单个 `results.csv`，列名清洗 + 数值化 + 按 epoch 排序                                   |
| `read_run_label`     | 从路径提取短图例名（如 `.../ghost/results.csv` → `ghost`；`root` 指定相对根）             |
| `last_n_stats`       | 末 N epoch 的 mean / std / best / n / start_epoch                                         |
| `smooth_series`      | 滑动平均平滑（`window <= 1` 原样返回）                                                    |
| `METRICS` / `Metric` | 指标注册表：15 个指标，分训练损失 / 验证指标 / 验证损失 / 训练过程四组                    |

### 🛠️ 通用文件操作 — `systools.py`

| 函数                   | 作用                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `create_dirs`          | 建目录，`overwrite` 可取 `None`（交互确认）/ `True` / `False` |
| `rm_dirs`              | 递归删除目录                                                  |
| `fetch_specific_files` | 按扩展名收集文件（支持传目录或已有路径列表）                  |
| `process_files`        | 批量复制（`mode=0`）或删除（`mode=1`）                        |
| `add_suffix_to_files`  | 给文件夹内文件批量加后缀（不改扩展名），目标已存在则跳过      |
| `ignore_something`     | 屏蔽 warnings / logging                                       |

---

## 快速上手

仓库根目录就是包目录，引用前要保证**它的父目录**在 `sys.path` 上（把 `ObjectDetection` 文件夹放进工程后，父目录通常就是工程根目录）：

```python
import sys
sys.path.insert(0, r"<ObjectDetection 的父目录>")   # 例如 D:/A/CODE

# 推荐：显式导入需要的函数
from ObjectDetection import extract_frames, count_labels, spilt_train_test, train

# 也可以（会触发全部子模块加载，见下方注意点）
from ObjectDetection import *
```

如果调用脚本就放在仓库根目录里，也可以绕过包 `__init__`、直接导入同级模块（单文件模式，见[注意点 1](#1-导入即全量加载依赖)）。`high_dpi_predict.py` 例外——它用包内相对导入，只能走包模式（见注意点 13）：

```python
from preprocess import count_labels
```

先统计一下手里的标签分布，心里有数再动手：

```python
from ObjectDetection import count_labels

stats = count_labels(
    "./datasets/桥梁病害/labels",
    names={0: "裂缝", 1: "剥落", 2: "钢筋外露锈蚀"},
)
print(stats["counts"])      # {'0:裂缝': 812, '1:剥落': 340, ...}
print(stats["labels"])      # 标注总数
```

---

## 典型工作流

### 1️⃣ 视频抽帧，扩充数据

```python
from ObjectDetection import extract_frames

result = extract_frames(
    input_path="./datasets/巡检视频",
    output_dir="./datasets/巡检视频/frames",
    every_n_frames=10,        # 每 10 帧取 1 张
    image_ext="jpg",
    image_quality=100,
    recursive=True,           # 递归搜索子文件夹
    flat_output=True,         # 所有图片放同一文件夹
    filename_prefix=True,     # 文件名带视频名前缀，防止重名覆盖
)
print(result["total_saved"])  # 提取图片总数
```

### 2️⃣ 标注格式统一成 YOLO txt

```python
from ObjectDetection import coco_to_txt, coco_txt_BaiDu, voc_to_txt

# 标准 COCO
coco_to_txt("./datasets/ann.json", "./datasets/labels", use_segments=False)

# 百度风格 COCO（annotations_path 传目录）
coco_txt_BaiDu("./datasets/baidu_ann/", "./datasets/labels")

# VOC XML（必须给类别映射）
voc_to_txt("./datasets/voc_xml/", "./datasets/labels",
           obj_dict={"crack": 0, "spalling": 1})
```

### 3️⃣ 标签治理：查漏、补全、统一类别

```python
from ObjectDetection import (
    check_and_copy_missing_files,
    replace_labels_with_target, batch_replace_labels,
    fill_empty_labels, normalize_labels,
)

# 找出落单文件，复制出来人工复核
check_and_copy_missing_files(
    img_path="./datasets/images",
    label_path="./datasets/labels",
    missing_dir="./datasets/_to_review",
)

# 把散乱的类别统一：1 和 2 都并成 0
replace_labels_with_target("./datasets/labels/a.txt", 1, 0)
replace_labels_with_target("./datasets/labels/a.txt", 2, 0, warn_if_no_match=True)

# 或整个文件夹批量（返回改动统计）
report = batch_replace_labels("./datasets/labels", origin_id=1, target_id=0,
                              warn_if_no_match=True)
print(report)   # {'total': 300, 'changed_files': 42, 'replaced_lines': 87, 'no_match_files': 5}

# 给没有目标的图补空标签（先 dry_run 预览）
fill_empty_labels("./datasets/images", "./datasets/labels", dry_run=True)
fill_empty_labels("./datasets/images", "./datasets/labels", dry_run=False)

# 夹紧越界坐标
normalize_labels("./datasets/labels")
```

### 4️⃣ 划分训练 / 验证集

```python
from ObjectDetection import spilt_train_test

spilt_train_test(
    li_path="./datasets/all",              # 混合存放图片与标签的目录，或路径列表
    train_img_path="./datasets/train/images",
    val_img_path="./datasets/val/images",
    train_label_path="./datasets/train/labels",
    val_label_path="./datasets/val/labels",
    val_size=0.2,                          # float=比例，int=数量
    random_state=110,                      # 固定种子，划分可复现
    negative_path="./datasets/negatives",  # 负样本混入训练集，压制误检
    upset_photo=True,                       # 先清空目标目录再划分
)
```

### 5️⃣ 训练 → 验证 → 推理 → 导出

```python
from ObjectDetection import train, evaluate, get_best_last_model, export_model
from ObjectDetection import advanced_predict, calc_complexity_yolo

# 训练
train(
    model_selection="yolo11n.pt",
    yaml_data="./datasets/data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,                 # -1 = 自动
    optimizer="Adam",
    cos_lr=True,
    seed_change=False,        # False 固定 seed=1，便于复现对比
    project="./runs/detect",
    mosaic=1.0,               # mosaic 增强概率（1.0 = 全开）
    close_mosaic=10,          # 最后 10 轮关闭 mosaic，让模型收敛到真实分布
)

# 验证
metrics = evaluate("./runs/detect/train/weights/best.pt", "./datasets/data.yaml")

# 定位权重
best = get_best_last_model(mode=0)      # mode=0 → best.pt；mode=1 → last.pt

# 复杂度统计（论文里的 Params / FLOPs）
stats = calc_complexity_yolo(best, imgsz=640)
print(stats["params_M"], stats["GFLOPs"])   # e.g. 2.590 8.700

# 推理（自绘框 + 中文标签）
advanced_predict(
    model_selection=best,
    img_path="./datasets/val/images",
    conf=0.5,
    font_path="C:/Windows/Fonts/simhei.ttf",       # 给字体路径才能显示中文
    replace_text={0: "裂缝", 1: "剥落"},            # 类别名整表替换
    save=True,                # 存到 ./results/
    show=False,               # 批量跑图务必置 False
)

# 导出 ONNX 部署
export_model(best, format="onnx")
```

### 6️⃣ 多实验指标对比，出论文配图

```python
from ObjectDetection import compare_metrics, list_metrics

# 先看有哪些指标可选，凭编号挑（返回 DataFrame，同时打印编号表）
list_metrics("./runs/detect/baseline/yolo11/results.csv")

result = compare_metrics(
    csv_paths=[                                  # 也可直接传目录，自动递归找 results.csv
        "./runs/detect/baseline/yolo11/results.csv",
        "./runs/detect/improvement/GhostConv/results.csv",
        "./runs/detect/improvement/GhostConv_SPPELAN_SimAM/results.csv",
    ],
    metrics=["mAP50-95", "mAP50", "box_loss"],  # 编号 / key / 列名 / 模糊子串都行
    labels=["Baseline", "GhostConv", "Ours"],   # 不给就用路径里的目录名
    last_n=10,          # 统计口径：末 10 个 epoch
    std_band=True,      # 叠加末 10 epoch 均值参考线
    smooth_window=3,    # 曲线滑动平均
    root="./runs/detect",   # 提取图例名时的相对根目录
    out_dir="figures", out_name="mAP_compare", dpi=300,
    show=False,
)
print(result["stats"]["mAP50-95"]["Ours"])   # {'mean':…, 'std':…, 'best':…, 'n':…, 'start_epoch':…}
```

### 7️⃣ 视觉大模型生成病害描述报告

检测框结果转 base64 data-url，交给本地 llama.cpp / vLLM 之类的 OpenAI 兼容视觉模型出文字报告。

```python
from ObjectDetection import chat_stream, getSystemPrompt

report = chat_stream(
    api_url="http://127.0.0.1:8080/v1/chat/completions",
    api_key="",                        # 本地服务一般留空
    model="Qwen2.5-VL-7B",
    provider="LLAMA.CPP",
    prompt="请描述图中的桥梁病害位置与严重程度。",
    image_prompt="data:image/jpeg;base64,...",
    systemPrompt=getSystemPrompt("./systemPrompt.txt"),
)
print(report)
```

需要流式逐字输出时改用类接口：

```python
from ObjectDetection import OpenAPI, getSystemPrompt

agent = OpenAPI("http://127.0.0.1:8080/v1/chat/completions", "", "Qwen2.5-VL-7B", "LLAMA.CPP")
gen = agent.openAPIRequest("描述病害", image_prompt=data_url,
                            systemPrompt=getSystemPrompt("./systemPrompt.txt"))
text = agent.parseGenerator(gen, show=True)   # show=True 同时逐块打印
```

### 8️⃣ 生成高 DPI 论文结果图

```python
# 未在 __init__.py 中 re-export，只能从子模块导入（前提见注意点 13）
from ObjectDetection.high_dpi_predict import predict_high_dpi

saved = predict_high_dpi(
    weights="./runs/detect/train/weights/best.pt",
    source="./datasets/val/images",   # 单图 / 目录（递归）/ 路径列表
    output_dir="./outputs/ours",
    conf=0.4, iou=0.7, imgsz=640,
    dpi=300,                          # 写入图片的 DPI 元数据
    font_weight="bold",               # regular / bold / italic / bold_italic 或字体路径
    font_size=None,                   # None = 按短边 4.2% 自适应
    box_width=None,                   # None = 按短边 / 180 自适应
    class_names={0: "crack"},
    scale=1.0,                        # >1 放大底图与框坐标（字号 / 线宽不跟着放大）
    save_format="png",
    show=False,
)
print(saved)
```

`dpi` 只写入元数据，不改变像素数；要真正提高分辨率用 `scale`。输出文件名为 `{原图名}.{save_format}`，同名会覆盖。

---

## 模块用法详解

### `count_labels` — 类别计数

统计标签分布是数据治理的第一步。支持 YOLO txt 与 VOC xml 混放，并可按格式分流查看。

```python
from ObjectDetection import count_labels

stats = count_labels(
    "./datasets/labels",
    recursive=True,                             # 含子文件夹
    label_exts=(".txt", ".xml"),
    names={0: "裂缝", 1: "剥落"},                 # ID → 名称，显示成 "0:裂缝"
    verbose=False,
)
```

返回结构：

| 键             | 含义                                                    |
| -------------- | ------------------------------------------------------- |
| `files`        | 扫描到的标签文件数                                      |
| `parsed_files` | 成功解析的文件数                                        |
| `labels`       | 标注总数                                                |
| `counts`       | `{类别: 数量}`，按数量降序                              |
| `by_format`    | `{扩展名: {类别: 数量}}`，只含出现过的格式              |
| `failed`       | `[(路径, 错误信息), ...]`，单文件解析失败不中断整体统计 |

> txt 统计出的是**类别 ID**（`"0"`），xml 统计出的是**类别名**（`"car"`）。混放时 `counts` 会同时出现两者，想分开看用 `by_format`。

### `fill_empty_labels` — 补齐空标签

YOLO 训练中「背景图」需要空的标签文件。本函数为每张缺失标签的图片生成 0 字节 `.txt`。

```python
from ObjectDetection import fill_empty_labels

report = fill_empty_labels(
    images_dir="./datasets/images",
    labels_dir="./datasets/labels",
    output_dir=None,          # None = 就地补全到 labels_dir
    recursive=False,
    label_ext=".txt",
    dry_run=True,             # ⭐ 先预览，确认后再改 False
    verbose=False,
)
# {'total': 1200, 'existed': 1180, 'created': 20, 'failed': 0, 'failures': []}
```

### `spilt_train_test` — 数据集划分

```python
spilt_train_test(
    li_path="./datasets/all",       # 目录 或 文件路径列表
    train_img_path=..., val_img_path=...,
    train_label_path=..., val_label_path=...,
    label_type="txt",
    negative_path=None,             # 负样本目录，随机混入训练集
    val_size=0.2,                   # float 比例 / int 数量
    random_state=110,
    upset_photo=False,              # True 会先清空目标目录
    need_test=False,                # 需同时 upset_photo=True
)
```

注意划分行为的两个坑：[见注意点 4](#4-数据集划分不校验图标签配对)。

### `predict` vs `advanced_predict`

|            | `predict`               | `advanced_predict`                  |
| ---------- | ----------------------- | ----------------------------------- |
| 绘图       | ultralytics 内置        | 自绘（OpenCV 矩形 + 文字背景块）    |
| 中文标签   | ❌ 不支持               | ✅ 传 `font_path` 即可              |
| 类别名替换 | 需改模型 `names`        | `replace_text={0: "裂缝"}` 整表替换 |
| 存图位置   | `runs/detect/predict*/` | `./results/`（相对当前工作目录）    |
| 适用场景   | 快速看效果、标准出图    | 交付中文可视化结果、论文配图        |

### `predict_high_dpi` — 高 DPI 结果图

| 参数                     | 默认              | 说明                                                                  |
| ------------------------ | ----------------- | --------------------------------------------------------------------- |
| `weights`                | `DEFAULT_WEIGHTS` | `.pt` 权重路径（默认值指向 CICTP 工程的绝对路径，见注意点 13）        |
| `source`                 | `DEFAULT_SOURCE`  | 单图 / 目录（递归）/ 路径列表                                         |
| `output_dir`             | `DEFAULT_OUTPUT`  | 输出目录，不存在会自动创建                                            |
| `conf` / `iou` / `imgsz` | 0.25 / 0.7 / 640  | 推理参数                                                              |
| `dpi`                    | 300               | 写入 PNG / JPEG 的 DPI 元数据，不改变像素数                           |
| `font_weight`            | `"bold"`          | `regular` / `bold` / `italic` / `bold_italic`，也可直接给字体文件路径 |
| `font_size`              | `None`            | None = `max(16, 短边 × 0.042)`                                        |
| `font_stroke`            | 0                 | 字体描边宽度，0 = 不描边                                              |
| `box_width`              | `None`            | None = `max(2, 短边 / 180)`                                           |
| `class_names`            | `None`            | `{类别ID: 显示名}`；未收录的 ID 回退模型自带名称                      |
| `scale`                  | 1.0               | 放大底图与框坐标；字号 / 线宽仍按原图短边计算                         |
| `save_format`            | `"png"`           | 输出扩展名；jpg / jpeg 走 quality=95，其它按 PNG 保存                 |
| `show`                   | False             | 每张图弹窗并阻塞等待按键                                              |

标签格式为 `{类别名} {置信度:.2f}`，优先贴在框上方，顶到图像边缘时改贴框内顶部；标签底色与框同色（默认蓝框白字）。

> 该模块未在 `__init__.py` 中 re-export，`from ObjectDetection import predict_high_dpi` 不可用；改用 `from ObjectDetection.high_dpi_predict import predict_high_dpi`，或用 `python -m ObjectDetection.high_dpi_predict` 运行内置示例（见注意点 13）。

### `export_model` — 模型导出

```python
from ObjectDetection import export_model

export_model("./runs/detect/train/weights/best.pt", format="onnx")
```

`format` 常用值：`onnx` / `torchscript` / `openvino` / `engine`（TensorRT，需匹配 CUDA 版本）。  
产物与权重**同目录、同名、不同后缀**。

### `extract_frames` — 视频抽帧

```python
from ObjectDetection import extract_frames

result = extract_frames(
    input_path="./datasets/clip.mp4",   # 文件 或 文件夹
    output_dir="./datasets/frames",
    every_n_frames=10,                  # 1 = 每帧都存
    image_ext="jpg",                    # jpg / jpeg / png / bmp / webp
    image_quality=100,                  # jpg、webp 生效；png 忽略
    recursive=True,
    flat_output=True,                   # False = 每个视频一个子文件夹
    filename_prefix=True,               # 图片名带视频名前缀
)
# {'results': [...], 'failures': [...], 'out_root': Path, 'total_saved': 350}
```

支持中文路径（内部走 `cv2.imencode` + `tofile`，规避 OpenCV 在 Windows 上的中文路径问题）。

### `systools` — 通用文件操作

```python
from ObjectDetection import create_dirs, rm_dirs, fetch_specific_files, process_files, add_suffix_to_files

create_dirs("./out", overwrite=False)          # None=交互确认 / True=清空重建 / False=保留
rm_dirs("./temp")
files = fetch_specific_files("./imgs", file_type=("jpg", "png"))
process_files("./src", "./dst", file_type="jpg", mode=0)   # mode=0 复制 / 1 删除
add_suffix_to_files("./imgs", "_v2")           # a.jpg → a_v2.jpg（目标已存在则跳过）
```

`fetch_specific_files` 传目录时内部用 `recursive=True`，会连子文件夹一起收集；传列表时按 `file_type` 过滤。

### `calc_complexity_yolo` — 模型复杂度统计

论文里报 Params / FLOPs 的标准口径（ultralytics 官方 `model.info()`）。

```python
from ObjectDetection import calc_complexity_yolo

r = calc_complexity_yolo("./runs/detect/train/weights/best.pt", imgsz=640, fuse=True)
print(r["params_M"], r["GFLOPs"])
```

| 参数              | 说明                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `model_selection` | `.pt` 权重路径                                                                                     |
| `imgsz`           | 输入尺寸，默认 640；会传给 `model.info()`，直接决定 GFLOPs 的计算分辨率                            |
| `fuse`            | 是否先做 Conv+BN 融合。**默认 True**，与验证 / 部署口径一致；设 False 则统计含独立 BN 的未融合模型 |

返回：`{"model", "imgsz", "fuse", "params_M", "GFLOPs"}`（保留 3 位小数）。

### `open_api` — 视觉大模型接口调用

封装 OpenAI 兼容的 `/v1/chat/completions`（本地 llama.cpp、vLLM、在线服务均可），支持**图文多模态**输入与**流式**输出。

```python
from ObjectDetection import OpenAPI, chat_stream, getSystemPrompt

# 方式一：函数式，一次拿完整文本
text = chat_stream(
    api_url="http://127.0.0.1:8080/v1/chat/completions",
    api_key="",
    model="Qwen2.5-VL-7B",
    provider="LLAMA.CPP",
    prompt="描述这张图里的病害。",
    image_prompt=data_url,               # 传 base64 data-url 走多模态分支
    systemPrompt=getSystemPrompt("./systemPrompt.txt"),
)

# 方式二：类接口，流式逐块拿
agent = OpenAPI(api_url, api_key, model, provider)
gen = agent.openAPIRequest(prompt, image_prompt=data_url, systemPrompt=sp)
text = agent.parseGenerator(gen, show=True)
```

`chat_stream` 与 `openAPIRequest` 还接受 `temperature` / `top_p` / `max_tokens` / `think` / `show`；`systemPrompt=None` 时使用 `DEFAULT_SYSTEM_PROMPT`。

| 成员                        | 说明                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `OpenAPI.__init__`          | 四个参数全为必填：`specific_api_url` / `specific_api_key` / `model` / `provider`，不做「留空即推导」 |
| `headerCreate()`            | 构造请求头；`api_key` 非空时才附带 `Authorization: Bearer ...`                                       |
| `playLoadCreate()`          | 组装 payload；`image_prompt` 为 `None` 走纯文本分支，否则走图文多模态分支                            |
| `openAPIRequest()`          | 发请求并**逐块 yield** 文本增量（生成器，请求在首次迭代时才真正发出）                                |
| `parseGenerator(gen, show)` | 把流式生成器收成完整字符串；`show=True` 同时逐块打印                                                 |
| `getSystemPrompt(path)`     | 读取 system prompt 文本文件                                                                          |

> `requests` 在本模块内是**惰性导入**——只有真正发请求时才 import，所以没装 `requests` 也不影响 `from ObjectDetection import ...` 其余功能，调用时会给出明确的 `ImportError` 提示。

### `compare_metrics` — 训练指标对比可视化

读 ultralytics 每次训练产出的 `results.csv`，把损失与验证指标画成曲线，支持多次实验叠加对比。适合记录消融实验、做论文配图。**只读 csv，不改动任何训练产物。**

```python
from ObjectDetection import compare_metrics

result = compare_metrics(
    csv_paths="./runs/detect/mosaic 10",   # 传目录会自动递归找 results.csv
    metrics=["mAP50-95", "mAP50"],         # 不给就是该 csv 里全部可用指标
    last_n=10,
    out_dir="figures",                     # 给了就落盘为 figures/metrics_compare.png
    show=False,
)
print(result["mode"])                      # 'single'（单次训练曲线）/ 'multi'（横向对比）
print(result["labels"])                    # 实际使用的图例名（重名会自动加 #2 后缀）
print(result["stats"]["mAP50-95"])
```

`root` 只影响图例名（相对它取父目录名），不改变实际读取的路径；传目录时 `recursive=True`（默认）会递归查找子目录里的 `results.csv`。

返回结构：

| 键          | 含义                                                     |
| ----------- | -------------------------------------------------------- |
| `mode`      | `single` / `multi`                                       |
| `csv_paths` | 实际读到的 `results.csv` 路径列表                        |
| `labels`    | 去重后的图例名                                           |
| `metrics`   | 解析出的 `Metric` 列表                                   |
| `stats`     | `{指标key: {图例名: {mean, std, best, n, start_epoch}}}` |
| `curve_fig` | matplotlib `Figure`，便于二次加工                        |

指标选择（`metrics`）支持以下写法：

| 写法                           | 例子                    |
| ------------------------------ | ----------------------- |
| 编号                           | `7` → mAP@0.5:0.95      |
| key                            | `"mAP50-95"`            |
| 完整列名                       | `"metrics/mAP50-95(B)"` |
| 模糊子串（忽略大小写与分隔符） | `"map50"`               |
| `Metric` 对象                  | `METRICS[6]`            |
| 全部可用                       | `None`                  |

绘图行为：1 个指标单列；2–4 个指标 2 列；5 个及以上 3 列。多文件叠加时用 figure 级统一图例；`std_band=True` 额外叠加末 N epoch 均值参考线；配色用 Okabe-Ito 无障碍色板，中文字体从 7 种候选中自动挑第一个已安装的。

---

## ⚠️ 危险操作清单

| 函数                                                  | 风险行为                                 | 是否可逆 |
| ----------------------------------------------------- | ---------------------------------------- | -------- |
| `rm_files`                                            | 物理删除图片**及其同名标签**，不进回收站 | ❌       |
| `rm_dirs`                                             | 递归删除整个目录                         | ❌       |
| `process_files(mode=1)`                               | 逐个物理删除源文件                       | ❌       |
| `create_dirs(overwrite=True)`                         | 清空并重建已存在的目录                   | ❌       |
| `spilt_train_test(upset_photo=True)`                  | 清空四个目标目录中的现有文件后再划分     | ❌       |
| `split_meta`                                          | 删除并重建 `./meta_split_results`        | ❌       |
| `coco_to_txt` / `coco_txt_BaiDu` / `voc_to_txt`       | `save_dir` 已存在则**整个删除重建**      | ❌       |
| `rm_icc_profile`                                      | 就地覆盖写 PNG                           | ❌       |
| `normalize_labels`                                    | 就地覆盖写 txt 坐标                      | ❌       |
| `replace_labels_with_target` / `batch_replace_labels` | 就地覆盖写标签类别                       | ❌       |
| `add_suffix_to_files`                                 | 就地重命名文件                           | ❌       |
| `predict_high_dpi` / `save_with_dpi`                  | 输出目录中同名图片会被覆盖               | ❌       |

**建议**：动手前先备份，尤其是 `rm_*` 系列与 `upset_photo=True`。

---
