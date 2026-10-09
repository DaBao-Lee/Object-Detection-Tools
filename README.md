# Object-Detection-Tools

一个**独立、通用的目标检测工程工具包**，基于 **Ultralytics YOLO**。不依赖任何特定项目，把 `tools/` 目录放进你的工程，用 `from tools import ...` 按需调用。覆盖从**视频抽帧 → 标注格式转换 → 标签治理 → 数据集划分 → 训练 → 验证 → 推理 → 模型导出 → 训练指标对比**的完整链路。

---

## 环境与依赖

Python ≥ 3.9（开发环境为 3.12 / 3.13）。

```bash
pip install ultralytics opencv-python numpy pillow tqdm lxml matplotlib pandas
```

| 依赖              | 用途                                   | 涉及模块                                       |
| --------------- | ------------------------------------ | ------------------------------------------ |
| `ultralytics`   | YOLO / YOLOWorld 训练、验证、推理、导出、COCO 转换 | `train` `val` `predict` `export` `convert` |
| `opencv-python` | 图像读写、画框、显示、视频解码                      | `predict` `video`                          |
| `numpy`         | 坐标数组运算                               | `predict` `convert`                        |
| `pillow`        | 中文文本绘制、ICC 配置处理、图片收集                 | `predict` `preprocess`                     |
| `tqdm`          | 进度条                                  | 全部模块                                       |
| `lxml`          | VOC XML 解析                           | `convert`                                  |
| `matplotlib`    | 训练曲线绘制、中文字体与论文白底样式           | `compare_metrics`                          |
| `pandas`        | `results.csv` 读取与统计计算                | `compare_metrics`                          |

---

## 目录结构

```
ObjectDetection/
├── main.py                  # 入口壳（仅 from tools import *，预留）
├── update.bat               # Windows 一键提交并推送脚本
└── tools/                   # 全部功能所在
    ├── __init__.py          # 统一 re-export + __all__
    ├── convert.py           # 标注格式转换：COCO / 百度 COCO / VOC → YOLO txt
    ├── preprocess.py        # 图像与标签预处理、标签类别治理、类别计数
    ├── dataset.py           # 图标签配对检查、数据集划分、分块收集
    ├── train.py             # YOLO 训练封装
    ├── val.py               # 模型验证
    ├── predict.py           # 推理（内置绘图 / 自绘中文标签两套）
    ├── export.py            # 模型导出（ONNX / TensorRT / OpenVINO …）
    ├── model_utils.py       # 训练产物定位（best.pt / last.pt）
    ├── compare_metrics.py   # 训练指标对比可视化（results.csv → 曲线 / 统计）
    ├── systools.py          # 通用文件操作（建目录、复制、删除、批量重命名）
    └── video.py             # 视频抽帧（单文件 / 批量 / 递归）
```

---

## 功能总览

### 🎬 数据获取 — `video.py`

| 函数               | 作用                        |
| ---------------- | ------------------------- |
| `extract_frames` | 主入口：批量抽帧，含配置打印、逐视频报告与汇总统计 |
| `process_video`  | 抽取单个视频的帧                  |
| `collect_videos` | 从文件或文件夹收集视频               |

### 🔄 标注转换 — `convert.py`

| 函数               | 作用                                                 |
| ---------------- | -------------------------------------------------- |
| `coco_to_txt`    | 标准 COCO json → YOLO txt（官方转换器，按 `cls91to80` 映射类别）  |
| `coco_txt_BaiDu` | 百度风格 COCO json → YOLO txt（自解析，不做类别映射，跳过 `iscrowd`） |
| `voc_to_txt`     | VOC XML → YOLO txt（需提供类别名→ID 映射字典）                 |

### 🧹 标签治理 — `preprocess.py`

| 函数                           | 作用                                                 |
| ---------------------------- | -------------------------------------------------- |
| `count_labels`               | 统计各类别标注数量，支持 YOLO txt + VOC xml，可配 `names` 显示中文类名  |
| `replace_labels_with_target` | 把标签文件里**指定类别**的 ID 改成目标类别（`origin_id="all"` 为全量替换） |
| `batch_replace_labels`       | 上者的文件夹批量版，返回改动统计                                   |
| `fill_empty_labels`          | 为缺失标签的图片生成空标签文件（YOLO 背景图），支持 `dry_run` 预览          |
| `normalize_labels`           | 把标签坐标夹紧到 1.0 以内                                    |
| `rm_icc_profile`             | 删除 PNG 的 ICC 色彩配置（部分标注工具对带 ICC 的图兼容性差）             |
| `collect_images`             | 收集目录下所有图片                                          |
| `IMAGE_EXTS`                 | 模块级常见图片扩展名集合                                       |

### 📂 数据集组织 — `dataset.py`

| 函数                             | 作用                                        |
| ------------------------------ | ----------------------------------------- |
| `check_and_copy_missing_files` | 检查图/标签配对，找出「有图无标签」「有标签无图」的落单样本            |
| `spilt_train_test`             | 划分 train/val（可选 test）并复制落盘，支持负样本混入与固定随机种子 |
| `split_meta`                   | 把目录按 `split_num` 平均切成若干份（分布式标注 / 分批处理用）   |
| `collcet_meta`                 | 收集 `split_meta` 各子目录的文件，可另存一份             |
| `copy_files`                   | 复制文件，同名跳过，可安全重跑                           |
| `rm_files`                     | 删除指定文件**及其同名标签**（保证配对不被破坏）                |

### 🏋️ 训练 / 验证 / 推理 / 导出

| 模块               | 函数                    | 作用                                                                   |
| ---------------- | --------------------- | -------------------------------------------------------------------- |
| `train.py`       | `train`               | YOLO 训练封装，暴露 epochs / batch / imgsz / optimizer / iou / mosaic 等常用超参 |
| `val.py`         | `evaluate`            | 在数据集上验证模型，返回 ultralytics 指标                                          |
| `predict.py`     | `predict`             | 用 ultralytics 内置绘图推理，可 `show` 弹窗 / `save` 存盘                         |
| `predict.py`     | `advanced_predict`    | **自绘检测框**，支持中文字体与类别名整表替换                                             |
| `predict.py`     | `add_pillow_text`     | 在图像数组上写文本（走 Pillow，解决 OpenCV 不支持中文）                                  |
| `predict.py`     | `get_infer_data`      | 按 `IMAGE`（随机抽样）/ `DIR`（整目录）取推理数据                                     |
| `export.py`      | `export_model`        | 导出 ONNX / TensorRT / OpenVINO / TorchScript                          |
| `model_utils.py` | `get_best_last_model` | 从训练产物中定位 `best.pt` / `last.pt`                                       |

### 📈 训练指标对比 — `compare_metrics.py`

| 函数 | 作用 |
|---|---|
| `compare_metrics` | **总入口**：读一个或多个 `results.csv`，出对比曲线 + 末 N epoch 统计 |
| `plot_curves` | 唯一出图函数（曲线叠加，可选末 N 均值参考线） |
| `list_metrics` | 罗列可选指标（编号 / key / 列名 / 趋势），打印编号表并返回 DataFrame |
| `resolve_metrics` | 把编号 / key / 完整列名 / 模糊子串解析成 `Metric` 列表 |
| `expand_csv_inputs` | str / Path / 目录 / 列表 → `results.csv` 路径列表（去重、保序） |
| `load_results_csv` | 读单个 `results.csv`，列名清洗 + 数值化 + 按 epoch 排序 |
| `read_run_label` | 从路径提取短图例名（如 `.../ghost/results.csv` → `ghost`） |
| `last_n_stats` | 末 N epoch 的 mean / std / best / n / start_epoch |
| `smooth_series` | 滑动平均平滑（`window <= 1` 原样返回） |
| `METRICS` / `Metric` | 指标注册表：15 个指标，分训练损失 / 验证指标 / 验证损失 / 训练过程四组 |

### 🛠️ 通用文件操作 — `systools.py`

| 函数                     | 作用                                                |
| ---------------------- | ------------------------------------------------- |
| `create_dirs`          | 建目录，`overwrite` 可取 `None`（交互确认）/ `True` / `False` |
| `rm_dirs`              | 递归删除目录                                            |
| `fetch_specific_files` | 按扩展名收集文件（支持传目录或已有路径列表）                            |
| `process_files`        | 批量复制（`mode=0`）或删除（`mode=1`）                       |
| `add_suffix_to_files`  | 给文件夹内文件批量加后缀（不改扩展名），目标已存在则跳过                      |
| `ignore_something`     | 屏蔽 warnings / logging                             |

---

## 快速上手

```python
# 推荐：显式导入需要的函数
from tools import extract_frames, count_labels, spilt_train_test, train

# 也可以（会触发全部子模块加载，见下方注意点）
from tools import *
```

先统计一下手里的标签分布，心里有数再动手：

```python
from tools import count_labels

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
from tools import extract_frames

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
from tools import coco_to_txt, coco_txt_BaiDu, voc_to_txt

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
from tools import (
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
from tools import spilt_train_test

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
from tools import train, evaluate, get_best_last_model, export_model
from tools import advanced_predict

# 训练
train(
    model_selection="yolov11n.pt",
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
from tools import compare_metrics, list_metrics

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

---

## 模块用法详解

### `count_labels` — 类别计数

统计标签分布是数据治理的第一步。支持 YOLO txt 与 VOC xml 混放，并可按格式分流查看。

```python
from tools import count_labels

stats = count_labels(
    "./datasets/labels",
    recursive=True,                             # 含子文件夹
    label_exts=(".txt", ".xml"),
    names={0: "裂缝", 1: "剥落"},                 # ID → 名称，显示成 "0:裂缝"
    verbose=False,
)
```

返回结构：

| 键              | 含义                                 |
| -------------- | ---------------------------------- |
| `files`        | 扫描到的标签文件数                          |
| `parsed_files` | 成功解析的文件数                           |
| `labels`       | 标注总数                               |
| `counts`       | `{类别: 数量}`，按数量降序                   |
| `by_format`    | `{扩展名: {类别: 数量}}`，只含出现过的格式         |
| `failed`       | `[(路径, 错误信息), ...]`，单文件解析失败不中断整体统计 |

> txt 统计出的是**类别 ID**（`"0"`），xml 统计出的是**类别名**（`"car"`）。混放时 `counts` 会同时出现两者，想分开看用 `by_format`。

### `fill_empty_labels` — 补齐空标签

YOLO 训练中「背景图」需要空的标签文件。本函数为每张缺失标签的图片生成 0 字节 `.txt`。

```python
from tools import fill_empty_labels

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

|       | `predict`               | `advanced_predict`            |
| ----- | ----------------------- | ----------------------------- |
| 绘图    | ultralytics 内置          | 自绘（OpenCV 矩形 + 文字背景块）         |
| 中文标签  | ❌ 不支持                   | ✅ 传 `font_path` 即可            |
| 类别名替换 | 需改模型 `names`            | `replace_text={0: "裂缝"}` 整表替换 |
| 存图位置  | `runs/detect/predict*/` | `./results/`（相对当前工作目录）        |
| 适用场景  | 快速看效果、标准出图              | 交付中文可视化结果、论文配图                |

### `export_model` — 模型导出

```python
from tools import export_model

export_model("./runs/detect/train/weights/best.pt", format="onnx")
```

`format` 常用值：`onnx` / `torchscript` / `openvino` / `engine`（TensorRT，需匹配 CUDA 版本）。  
产物与权重**同目录、同名、不同后缀**。

### `extract_frames` — 视频抽帧

```python
from tools import extract_frames

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
from tools import create_dirs, rm_dirs, fetch_specific_files, process_files, add_suffix_to_files

create_dirs("./out", overwrite=False)          # None=交互确认 / True=清空重建 / False=保留
rm_dirs("./temp")
files = fetch_specific_files("./imgs", file_type=("jpg", "png"))
process_files("./src", "./dst", file_type="jpg", mode=0)   # mode=0 复制 / 1 删除
add_suffix_to_files("./imgs", "_v2")           # a.jpg → a_v2.jpg（目标已存在则跳过）
```

### `compare_metrics` — 训练指标对比可视化

读 ultralytics 每次训练产出的 `results.csv`，把损失与验证指标画成曲线，支持多次实验叠加对比。适合记录消融实验、做论文配图。**只读 csv，不改动任何训练产物。**

```python
from tools import compare_metrics

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

返回结构：

| 键 | 含义 |
|---|---|
| `mode` | `single` / `multi` |
| `csv_paths` | 实际读到的 `results.csv` 路径列表 |
| `labels` | 去重后的图例名 |
| `metrics` | 解析出的 `Metric` 列表 |
| `stats` | `{指标key: {图例名: {mean, std, best, n, start_epoch}}}` |
| `curve_fig` | matplotlib `Figure`，便于二次加工 |

指标选择（`metrics`）支持四种写法：

| 写法 | 例子 |
|---|---|
| 编号 | `7` → mAP@0.5:0.95 |
| key | `"mAP50-95"` |
| 完整列名 | `"metrics/mAP50-95(B)"` |
| 模糊子串（忽略大小写与分隔符） | `"map50"` |
| 全部可用 | `None` |

绘图行为：1 个指标单图，多指标自动排成子图网格（`1 / 2 / 3` 列）；多文件叠加时用 figure 级统一图例；`std_band=True` 额外叠加末 N epoch 均值参考线；配色用 Okabe-Ito 无障碍色板，中文字体从 8 种候选中自动挑第一个已安装的。

---

## ⚠️ 危险操作清单

| 函数 | 风险行为 | 是否可逆 |
|---|---|---|
| `rm_files` | 物理删除图片**及其同名标签**，不进回收站 | ❌ |
| `rm_dirs` | 递归删除整个目录 | ❌ |
| `process_files(mode=1)` | 逐个物理删除源文件 | ❌ |
| `create_dirs(overwrite=True)` | 清空并重建已存在的目录 | ❌ |
| `spilt_train_test(upset_photo=True)` | 清空四个目标目录中的现有文件后再划分 | ❌ |
| `split_meta` | 删除并重建 `./meta_split_results` | ❌ |
| `coco_to_txt` / `coco_txt_BaiDu` / `voc_to_txt` | `save_dir` 已存在则**整个删除重建** | ❌ |
| `rm_icc_profile` | 就地覆盖写 PNG | ❌ |
| `normalize_labels` | 就地覆盖写 txt 坐标 | ❌ |
| `replace_labels_with_target` / `batch_replace_labels` | 就地覆盖写标签类别 | ❌ |
| `add_suffix_to_files` | 就地重命名文件 | ❌ |

**建议**：动手前先备份，尤其是 `rm_*` 系列与 `upset_photo=True`。

---

## 已知注意点

### 1. 导入即全量加载依赖

`tools/__init__.py` 会 import 全部子模块，因此**只要装了 `tools` 包就要求所有依赖齐备**。仅想用轻量函数（如 `count_labels`，只依赖 PIL + tqdm）时，可绕过包 `__init__` 直接导入子模块：

```python
from tools.preprocess import count_labels      # 不需要 ultralytics / cv2
```

### 2. 输出路径是相对当前工作目录的

以下路径**不是**相对被处理文件，而是相对**调用时的 CWD**：

| 函数 | 输出位置 |
|---|---|
| `split_meta` | `./meta_split_results/{split_name}{i}/` |
| `collcet_meta(save=True)` | `./meta` |
| `advanced_predict(save=True)` | `./results/` |
| `predict(save=True)` | `runs/detect/predict*/`（ultralytics 约定） |
| `get_best_last_model` | 默认 `./runs/detect/train/weights` |

在别的目录调用脚本时会写到意料之外的地方——建议在项目根目录执行，或改用绝对路径。

### 3. `need_test=True` 的两个前提

该分支固定**从训练集随机抽 100 张**作测试集，因此要求：

1. 必须同时设 `upset_photo=True`（否则不会触发清空分支）；
2. 训练集样本数需 ≥ 100，否则 `random.sample` 报错。

### 4. 数据集划分**不校验**图标签配对

`spilt_train_test` 把图片与标签**各自独立划分**，只按文件名主干在 val 集合内做匹配，并不检查图与标签是否成对。因此：

- 有图无标签 → 这张图会照常进入 train/val，但训练时该样本无标注；
- 有标签无图 → 该标签文件也会被复制过去，成为孤儿标签。

**建议先跑一遍配对检查再划分**（或传 `negative_path` 之外自行清洗）：

```python
from tools import check_and_copy_missing_files, spilt_train_test

check_and_copy_missing_files(
    "./datasets/all", "./datasets/all",
    missing_dir="./datasets/_to_review",   # 落单文件复制出来人工复核
)
spilt_train_test(...)
```

另外 `negative_path` 只识别 `*.jpg`，且抽样数有硬上限——**最多 2500 张，且不超过训练集规模的 10%**；若该目录的 jpg 数量不足抽样数会抛 `ValueError`。

### 5. `collcet_meta` 是既有拼写

函数名 `collcet_meta`（应为 `collect`）是历史命名，作为公开 API 保留以免破坏调用方。它不是笔误的当前状态，而是既定接口。

### 6. 已废弃 / 保留参数

- `copy_files(verbose=...)`：为兼容旧调用保留，当前不产生输出。
- `normalize_labels` 内部用 `text.index(tt)` 定位待改写列。若某行的**类别 ID 字符串恰好与某个 ≥1.0 的坐标值相同**（如一行形如 `1 1 0.5 0.2 0.1`），`index()` 会命中第 0 列的类别 ID 并被改写成 `0.99999`，导致该行类别被破坏。代码注释中已标注此风险，批量使用前建议先抽查。

### 7. `compare_metrics` 的统计口径是「末 N 个 epoch」

`last_n` 默认 10 而**不是全程**，因为 `close_mosaic` 之前的指标仍受 mosaic 增强扰动、波动很大，取末 N 轮更接近收敛值。`last_n=None` 会统计全程，**不推荐**——这也正是 `train()` 里 `close_mosaic=10` 与本模块默认口径能对上的原因。

另外 `metrics=None` 会把该 csv 里所有可用指标全画出来（注册表共 15 个），子图网格按指标数自动排 1 / 2 / 3 列，指标多时图会偏大。

### 8. `compare_metrics.py` 可以单独运行

文件底部有 `if __name__ == "__main__"` 的 6 个用例（单文件曲线、多文件叠加、递归目录、编号选指标等），但里面的 `runs/detect/mosaic 10` 等路径是**硬编码的演示路径**，需自备 `results.csv` 才能跑通。该分支还依赖包名 `ObjectDetection` 可导入——若你把 `tools/` 改名或单独拷走，直接运行该文件会报导入错误，此时改用 `from tools.compare_metrics import compare_metrics` 调用即可。

---