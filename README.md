# ObjectDetection

一个**独立、通用的目标检测工程工具包**，基于 **Ultralytics YOLO**。仓库根目录本身就是一个 Python 包（`ObjectDetection`）：把**它的父目录**加入 `sys.path`，即可用 `from ObjectDetection import ...` 按需调用，不依赖任何特定项目。覆盖从**视频抽帧 → 标注格式转换 → 标签治理 → 数据集划分 → 训练 → 验证 → 推理 / 高 DPI 出图 / 视频流单帧推理与渲染 → 模型导出 → 复杂度统计 → 视觉大模型报告**的完整链路。

---

## 环境与依赖

Python ≥ 3.10（`preprocess.py` 使用 `set | None` 注解，3.9 导入即抛 `TypeError`；开发环境为 3.12 / 3.13）。

```bash
pip install ultralytics opencv-python numpy pillow tqdm lxml requests

# 只在要用 .onnx 引擎时额外装（predict.py 里的 onnx_predict 需要；不装不影响其余模块与整包导入）
pip install onnxruntime-gpu==1.20.0   # 导入名仍是 import onnxruntime；只有 -gpu 包带 CUDAExecutionProvider
```

| 依赖            | 用途                                               | 涉及模块                                                                             |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `ultralytics`   | YOLO 训练、验证、推理、导出、COCO 转换             | `train` `val`（含 `calc_complexity_yolo`）`predict` `high_dpi_predict` `export` `convert` |
| `opencv-python` | 图像读写、画框、显示、视频解码                     | `predict` `draw_results` `high_dpi_predict` `video_process`                          |
| `numpy`         | 坐标数组运算、前处理张量                           | `predict` `draw_results` `high_dpi_predict` `convert` `video_process` |
| `pillow`        | 中文文本绘制、标签排版、ICC 配置处理、图片收集     | `predict` `draw_results` `high_dpi_predict` `preprocess`                             |
| `tqdm`          | 进度条                                             | `convert` `preprocess` `dataset` `systools` `video_process`（ultralytics 内部也用）   |
| `lxml`          | VOC XML 解析                                       | `convert`                                                                            |
| `requests`      | 调用 OpenAI 兼容的视觉大模型接口                   | `open_api`（**惰性导入**，不装也不影响其它模块）                                     |
| `onnxruntime-gpu` | .onnx 单帧推理（CUDA / CPU 双 provider）         | `predict` 里的 `.onnx` 引擎（`predict.py` 用 `try/except` 导入它，不装也能整包导入，只有调 `onnx_predict` 才报错） |

> `onnxruntime-gpu` 有两个前提，缺一个就静默退回 CPU、耗时翻倍：① 环境里已有 `torch`（onnxruntime 的 cublas/cudnn DLL 靠它带进来）；② 代码里 `import torch` 必须排在 `import onnxruntime` 前面——`predict.py` 顶部已按这个顺序写好，别调换。自检：`python -c "import torch, onnxruntime as o; print(o.get_available_providers())"`，列表里出现 `CUDAExecutionProvider` 才算通。

---

## 目录结构

```
ObjectDetection/                 # 仓库根目录即 Python 包
├── __init__.py                  # 统一 re-export + __all__（63 个符号，不含 high_dpi_predict）
├── convert.py                   # 标注格式转换：COCO / 百度 COCO / VOC → YOLO txt
├── preprocess.py                # 图像与标签预处理、标签类别治理、类别计数
├── dataset.py                   # 图标签配对检查、数据集划分、分块收集
├── train.py                     # YOLO 训练封装
├── val.py                       # 模型验证 + 复杂度统计（evaluate / calc_complexity_yolo）
├── predict.py                   # 推理三合一：批量图片（predict）+ .pt 单帧引擎 + .onnx 单帧引擎
├── draw_results.py              # 检测结果渲染（画框 + 标签、日志面板、窗口居中、组装 PredictResult）
├── high_dpi_predict.py          # 高 DPI 结果图（Times New Roman 标签 / DPI 元数据 / 放大输出）
├── export.py                    # 模型导出（ONNX / TensorRT / OpenVINO …）
├── model_utils.py               # 权重定位（best.pt / last.pt、按文件夹列权重）
├── systools.py                  # 通用文件操作（建目录、复制、删除、批量重命名）
├── video_process.py             # 视频抽帧存盘（单文件 / 批量 / 递归）+ 内存取帧（逐帧 / 跳帧）
├── open_api.py                  # 视觉大模型接口调用（OpenAI 兼容 / 流式 / 多模态）
├── README.md
└── update.bat                   # Windows 一键提交并推送脚本
```

---

## 功能总览

### 🎬 数据获取与取帧 — `video_process.py`

| 函数             | 作用                                                                       |
| ---------------- | -------------------------------------------------------------------------- |
| `extract_frames` | 主入口：批量抽帧**存盘**，含配置打印、逐视频报告与汇总统计                 |
| `process_video`  | 抽取单个视频的帧                                                           |
| `collect_videos` | 从文件或文件夹收集视频（按 `VIDEO_EXTS` 过滤）                             |
| `video_info`     | 读 fps / 宽 / 高 / 帧数，给分帧循环和前端进度当基准                        |
| `iter_frames`    | **内存逐帧**产出 `(帧号, BGR 帧)`，`stride` 抽帧、`max_frames` 截断        |
| `read_frame`     | 跳到指定帧，返回 `(实际帧号, BGR 帧)`；可复用已打开的 `cap` 省掉重开销     |
| `VIDEO_EXTS`     | 支持的视频扩展名集合（10 个，收集过滤与上传白名单共用）                    |

> `video_info` / `iter_frames` / `read_frame` 不解盘，直接给 `ndarray`，接 `pt_predict` / `onnx_predict` 用；`extract_frames` 是"生产数据集"那条线，两者不要混用。

### 🔄 标注转换 — `convert.py`

| 函数             | 作用                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `coco_to_txt`    | 标准 COCO json → YOLO txt（官方转换器，按 `cls91to80` 映射类别）      |
| `coco_txt_BaiDu` | 百度风格 COCO json → YOLO txt（自解析，不做类别映射，跳过 `iscrowd`） |
| `voc_to_txt`     | VOC XML → YOLO txt（需提供类别名→ID 映射字典）                        |

### 🧹 标签治理 — `preprocess.py`

| 函数                         | 作用                                                                       |
| ---------------------------- | -------------------------------------------------------------------------- |
| `count_labels`               | 统计各类别标注数量，支持 YOLO txt + VOC xml；`names` 可传 `{id: 名字}` 字典**或名称列表**，列表按顺序补下标 |
| `replace_labels_with_target` | 把标签文件里**指定类别**的 ID 改成目标类别（`origin_id="all"` 为全量替换）；`warn_if_no_match=True` 时命中 0 行告警 |
| `batch_replace_labels`       | 上者的文件夹批量版，返回改动统计 + `failed` 失败明细；非法 ID 在扫描前就报错 |
| `fill_empty_labels`          | 为缺失标签的图片生成空标签文件（YOLO 背景图），支持 `dry_run` 预览         |
| `normalize_labels`           | 标签坐标双向钳位（`>1.0` 与 `<0.0` 都改），只重写真正被改到的那一行；NaN 只上报为 `nan_values`，不写文件 |
| `rm_icc_profile`             | 删除 PNG 的 ICC 色彩配置（部分标注工具对带 ICC 的图兼容性差）；单张读不了不再中断整批，失败记在 `failures` |
| `collect_images`             | 收集目录下所有图片；`root` 收 `str` 或 `PathLike`，目录不存在抛 `FileNotFoundError`，`image_exts` 可覆盖默认后缀集合 |
| `IMAGE_EXTS`                 | 模块级常见图片扩展名集合                                                   |

### 📂 数据集组织 — `dataset.py`

| 函数                           | 作用                                                                                       |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| `check_and_copy_missing_files` | 只查一个方向：图片多则找缺标签的图，标签多则找缺图的标签；`meta_path` 再比对一份元数据图片；缺失目录自动创建 |
| `split_train_test`             | 划分 train/val（可选 test）并复制落盘，支持负样本混入与固定随机种子；`upset_photo=True` 删旧文件前逐条确认（`assume_yes=True` 跳过）；原拼错名 `spilt_train_test` 已废弃删除 |
| `split_meta`                   | 把目录按 `split_num` 均分成若干份（余数摊给前几段，不产生空份）；图标签数量不一致时告警并返回 `[]`；返回各份路径列表 |
| `collect_meta`                 | 收集 `split_meta` 各子目录的文件；`save=True` 另复制一份到 `./meta`，目录已存在不再报错；无数据时返回 `[]` |
| `copy_files`                   | 复制文件，同名跳过，目标目录自动创建；`max_num` 随机抽样，默认交互确认（`assume_yes=True` 跳过）；抽样数超过候选数给出明确报错 |
| `rm_files`                     | 删除指定文件**及其同名标签**（`.jpg`/`.png`/`.jpeg` 同名一起找，保证配对不被破坏）；打印删了什么，没删到给 `[warn]`，返回实际删除路径列表 |

### 🏋️ 训练 / 验证 / 推理 / 导出

| 模块                      | 函数                   | 作用                                                                                                                                                              |
| ------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.py`                | `train`                | YOLO 训练封装：`epochs` / `batch` / `imgsz` / `optimizer` / `lr` / `patience` / `mosaic` / `close_mosaic` / `seed_change`（False 固定 `seed=1`）等             |
| `val.py`                  | `evaluate`             | 验证：`iou`（NMS 阈值）/ `batch` / `rect` / `imgsz` / `conf`（默认 0.001），返回 ultralytics 验证结果；`rect` 口径见[注意点 9](#9-验证的-rect-口径不可混用)       |
| `predict.py`              | `predict`              | 用 ultralytics 内置绘图推理：`conf` / `iou` / `save` / `show` / `stream` / `verbose`                                                                                |
| `predict.py`              | `add_pillow_text`      | 在图像数组上写文本（走 Pillow，解决 OpenCV 不支持中文）                                                                                                           |
| `predict.py`              | `get_infer_data`       | `typing=InferDataType.IMAGE`（随机抽样，`max_num` 上限）/ `InferDataType.DIR`（整目录）；`recursive` 控制子目录；扩展名规则见注意点 11                            |
| `export.py`               | `export_model`         | 导出 ONNX / TensorRT / OpenVINO / TorchScript                                                                                                                      |
| `model_utils.py`          | `get_best_last_model`  | 定位 `best.pt` / `last.pt`；`index` 指定 `./runs/detect/train{index}/weights`                                                                                     |
| `model_utils.py`          | `list_weights`         | 列出一个文件夹里的权重（默认 `.pt` / `.onnx`），按文件名排序；`recursive` 递归、`exts` 传 `"pt"` 或 `".pt"` 都认                                                  |
| `val.py`                  | `calc_complexity_yolo` | 统计参数量（M）与计算量（GFLOPs），`fuse` 控制是否 BN 折叠，`imgsz` 参与 GFLOPs 计算                                                                              |

### ⚡ 单帧流式推理 — `predict.py` 的两个引擎

| 函数              | 作用                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| `pt_predict`      | `.pt` 单帧入口：模型按绝对路径缓存复用，出框后交给 `finish_predict`；参数与 `onnx_predict` 一一对应 |
| `warmup_pt`       | 全黑假帧走一遍 predict，提前掉 CUDA 初始化与 kernel 首编译（首次加载时已自动调一次）              |
| `onnx_predict`    | `.onnx` 单帧入口：自带 letterbox / 归一化 / 解码 / 逐类 NMS；session 按「路径 + providers」缓存   |
| `letterbox`       | 等比缩放后居中贴进 `size×size` 的 114 灰画布，返回 `(画布, scale, pad_x, pad_y)`                 |
| `nms_per_class`   | 逐类 NMS，返回保留框在原候选数组里的下标；按类分组是为了跨类不互相压制                            |

> 五个符号都在 `__init__.py` 里，直接 `from ObjectDetection import pt_predict, onnx_predict, letterbox, nms_per_class`。`predict.py` 顶部对 `onnxruntime` 做了 `try/except` 兜底：没装也能整包导入、也能正常用 `.pt` 引擎，只有真去调 `onnx_predict` / `_load_onnx_model` 时才抛 `ImportError` 提示装包。

> 两条链路出框后的画图口径完全一致，但 `infer_ms` 的**计时范围不同**：`.pt` 含 ultralytics 内部的预处理 + 推理 + NMS，`.onnx` 含 letterbox + 归一化 + `session.run`、**不含**它自己那步 Python 解码 / NMS。两者不要横向比大小，只跟同一引擎的另一次运行比。

### 🖌 结果渲染 — `draw_results.py`

| 函数                            | 作用                                                                             |
| ------------------------------- | -------------------------------------------------------------------------------- |
| `finish_predict`                | 统一收口：统计、画框、打面板、缩放、弹窗、组装 `PredictResult`（各引擎共用这一条） |
| `draw_boxes`                    | 在帧副本上画框 + 「类别 置信度」标签，输出仍是 BGR；标签贴不下自动改挂框内        |
| `draw_log_panel`                | 左上角叠半透明底，逐行写等宽对齐的「键 值」日志                                   |
| `load_font`                     | 字重名（`FONT_FILES` 的键）或 ttf 路径 → Pillow 字体；缺文件退默认字体且只警告一次 |
| `get_screen_size` / `center_window` | 读物理分辨率 / 把窗口摆到屏幕正中，同一尺寸只真搬一次                            |
| `Detection` / `PredictResult`   | 单个框与整帧产出的规范结构，各条链路都归一到这两个类型                            |
| `BOX_COLOR` … `PANEL_ALPHA`     | 渲染默认值：颜色统一按 BGR 收，需要 Pillow 时内部翻 RGB                           |

> `center_window` 扣掉的窗口装饰边（左边框 9px、标题栏 38px）是 Win 26300 + OpenCV 5.0 + 125% 缩放的实测值，换系统或换 OpenCV 版本可能要重新标定；搬不动（headless 等）返回 `(-1, -1)`，不抛异常。

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
from ObjectDetection import iter_frames, pt_predict, count_labels, split_train_test, train

# 也可以（会触发全部子模块加载，见下方注意点）
from ObjectDetection import *
```

如果调用脚本就放在仓库根目录里，也可以绕过包 `__init__`、直接导入同级模块（单文件模式，见[注意点 1](#1-导入即全量加载依赖)）。`high_dpi_predict.py` 例外——它用包内相对导入，只能走包模式（`draw_results.py` 自包含，同级直接 `from draw_results import ...` 也认；见注意点 13）：

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

# 或整个文件夹批量（非法 ID 在扫描前就报错；返回改动统计 + 失败明细）
report = batch_replace_labels("./datasets/labels", origin_id=1, target_id=0,
                              warn_if_no_match=True)
print(report)   # {'total': 300, 'changed_files': 42, 'replaced_lines': 87, 'no_match_files': 5, 'failed': []}

# 给没有目标的图补空标签（先 dry_run 预览）
fill_empty_labels("./datasets/images", "./datasets/labels", dry_run=True)
fill_empty_labels("./datasets/images", "./datasets/labels", dry_run=False)

# 钳位越界坐标（>1.0 与 <0.0 都修），只重写改到的那一行，未改动的行原样保留
print(normalize_labels("./datasets/labels"))
# {'files': 300, 'changed_files': 3, 'clipped': 7, 'nan_values': []}
```

### 4️⃣ 划分训练 / 验证集

```python
from ObjectDetection import split_train_test

split_train_test(
    li_path="./datasets/all",              # 混合存放图片与标签的目录，或路径列表
    train_img_path="./datasets/train/images",
    val_img_path="./datasets/val/images",
    train_label_path="./datasets/train/labels",
    val_label_path="./datasets/val/labels",
    val_size=0.2,                          # float=比例，int=数量
    random_state=110,                      # 固定种子，划分可复现
    negative_path="./datasets/negatives",  # 负样本混入训练集，压制误检
    upset_photo=True,                      # 先清空目标目录再划分（清空前会打印数量并等 y 确认）
    assume_yes=True,                       # 无人值守脚本用 True 跳过确认
)
```

### 5️⃣ 训练 → 验证 → 推理 → 导出

```python
from ObjectDetection import train, evaluate, get_best_last_model, export_model
from ObjectDetection import predict, calc_complexity_yolo

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

# 推理（批量跑图：ultralytics 内置绘图，save=True 存到 runs/detect/predict*/）
predict(
    model_selection=best,
    img_path="./datasets/val/images",
    conf=0.5,
    iou=0.7,
    save=True,
    show=False,               # 批量跑图务必置 False，否则每张图都阻塞等按键
)

# 要中文标签 / 自定义面板，改走单帧引擎逐帧处理（见 8️⃣）
# from ObjectDetection import pt_predict
# res = pt_predict(best, frame, font="bold", panel=True)

# 导出 ONNX 部署
export_model(best, format="onnx")
```

### 6️⃣ 视觉大模型生成病害描述报告

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

### 7️⃣ 生成高 DPI 论文结果图

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

### 8️⃣ 视频流单帧推理（网页 / 弹窗显示）

```python
from ObjectDetection import iter_frames, pt_predict, onnx_predict, read_frame, video_info

SRC = "./videos/2.mp4"
W = "./weights/yolov8n.pt"
ONNX = "./weights/yolov8n.onnx"

# 整段视频：逐帧进内存，画完框左上角自带日志面板
for idx, frame in iter_frames(SRC, stride=1):
    r = pt_predict(W, frame, conf=0.5)                 # 模型按路径缓存，只在第一帧加载
    print(r.count, r.counts_by_class, r.infer_ms, r.elapsed_ms)
    cv2.imshow("frame", r.annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 同一帧换引擎出图口径完全一致（耗时口径见上面 ⚡ 节的说明）
r2 = onnx_predict(ONNX, frame, conf=0.5, providers=["CPUExecutionProvider"])

# 前端拖进度条：只预测点到的那一帧，复用同一个 cap 句柄每帧省掉约 40ms 开容器
cap = cv2.VideoCapture(SRC)
real_idx, one = read_frame(SRC, int(video_info(SRC)["frames"] * 0.6), cap=cap)
r3 = pt_predict(W, one, conf=0.5, display_size=(640, 360), show_panel=False)
cv2.imwrite("./out.jpg", r3.display)                   # display 是成品图缩放版，xyxy 仍是原图坐标
cap.release()
```

首帧特别慢是正常的：`pt_predict` 第一次建模型约 1s、`.onnx` 第一次建 session 更久，预热已在加载时自动做一次（实测 `pt` 首帧 3441 ms → 218 ms）。给网页端用就把 `show=False`、`display_size` 设成推流尺寸，取 `r.display` 编码成 JPEG。

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

### `split_train_test` — 数据集划分

```python
split_train_test(
    li_path="./datasets/all",       # 目录 或 文件路径列表
    train_img_path=..., val_img_path=...,
    train_label_path=..., val_label_path=...,
    label_type="txt",
    negative_path=None,             # 负样本目录，随机混入训练集
    val_size=0.2,                   # float 比例 / int 数量
    random_state=110,
    upset_photo=False,              # True 会先清空目标目录（清空前逐条确认，除非 assume_yes=True）
    need_test=False,                # 需同时 upset_photo=True
    test_img_path=None,             # 不给则自动从 …/train_img 推出 …/test_img
    test_label_path=None,           # 同上
    test_size=100,                  # 测试集抽样数量
    assume_yes=False,               # True = 跳过所有交互确认
)
```

函数名原为 `spilt_train_test`（拼写有误），已改正并删除旧名，不再保留别名：调用方一律改用 `split_train_test`，仍写旧名会在 `from ObjectDetection import spilt_train_test` 处直接 `ImportError`。

### `pt_predict` / `onnx_predict` — 单帧流式推理

两个入口的参数**一一对应**，只差引擎侧那一项：`.pt` 用 `imgsz` / `device`，`.onnx` 用 `providers`（边长由输入节点 shape 自己决定）。

| 分组 | 参数                                     | 默认                            | 说明                                                                             |
| ---- | ---------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------- |
| 定位 | `weights`                                | 必填                            | 路径，或已经建好的 `YOLO` / `InferenceSession` 实例（原样透传，不再新建）       |
|      | `frame`                                  | 必填                            | BGR `ndarray`；返回的 `orig` 是它的副本，调用方手里的帧不会被改动                |
| 推理 | `conf` / `iou`                           | 0.25 / 0.7                      | 置信度阈值 / NMS IoU 阈值                                                        |
|      | `imgsz`（仅 pt）/ `device`（仅 pt）      | 640 / None                      | 推理边长；`device` 按 ultralytics 口径，如 `"0"` / `"cpu"`                      |
|      | `providers`（仅 onnx）                   | None                            | 默认探测本机：有 CUDA 就 `CUDA → CPU` 兜底，没有就纯 CPU                        |
|      | `class_names`                            | None                            | `{id: 名字}`；不传用权重自带 metadata，都没有退成数字 id 字符串                  |
| 画图 | `box_color` / `box_width` / `label_bg` / `text_color` / `font` / `font_size` / `font_stroke` | 红框(BGR) / 自适应 / 随框色 / 白 / `bold` / 自适应 / 0 | 自适应基准是输出图幅短边：字号 `max(16, 短边×4.2%)`、线宽 `max(2, 短边/180)` |
| 面板 | `draw` / `show` / `show_panel` / `panel_font_size` | True / False / True / 自适应 | 面板字号自适应 `max(14, 短边×3.6%)`；行序 = 总数 → 各类别 → `infer_ms` → 起止时刻 |
| 显示 | `display_size`                           | None                            | `(宽, 高)`，只影响 `result.display` 与弹窗；`Detection.xyxy` 仍是原图坐标         |

返回值 `PredictResult`：`infer_ms` 是引擎自己那段，`elapsed_ms` 是整帧（含画框与标签，不含画面板和弹窗）；时长走 `perf_counter`、时刻走 `time.time`，系统对时不会把耗时拨成负数。

`warmup_pt` 可以手动再压一遍（首次加载已自动压一次）；`letterbox` / `nms_per_class` 是 `.onnx` 那条线上的两个通用件，不想整套用 `onnx_predict` 时可以单拿。

> `draw_results.py` 和 `high_dpi_predict.py` 各有一套画框代码，**互不调用**：前者服务流式推理（有日志面板、`label_bg` 可独立设、坐标钳到 `w-1`、画在副本上），后者服务论文出图（标签底色恒等于框色、原地改画布、坐标钳到 `w`、写 DPI 元数据）。两套口径不同是刻意的，别当重复代码合并。

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

### `iter_frames` / `read_frame` / `video_info` — 内存取帧

```python
from ObjectDetection import iter_frames, read_frame, video_info, VIDEO_EXTS

video_info("./videos/2.mp4")
# {'fps': 30.0, 'width': 640, 'height': 360, 'frames': 210}   fps 拿到 0 的容器退成 25

for idx, frame in iter_frames("./videos/2.mp4", stride=5, max_frames=50):
    ...                       # 读到结尾自然停；调用方 break / 抛错时句柄也会被 release

idx, frame = read_frame("./videos/2.mp4", 999999)   # 越界钳到末帧，不报错
```

三个坑：① `read_frame` 回的是**实际**帧号，不是入参——个别容器 seek 会落到邻近帧；② 传 `cap=` 时复用调用方已打开的句柄，函数内部**不重开也不 release**，逐帧端点每帧能省掉约 40 ms；③ `frame_idx < 0` 抛 `ValueError`，容器不给帧数（`frames` 拿不到）时不做钳制。

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

论文里报 Params / FLOPs 的标准口径（ultralytics 官方 `model.info()`）。原独立文件 `calc_complexity_yolo.py` 已并入 `val.py`，导入方式不变（`from ObjectDetection import calc_complexity_yolo`）。

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

---

## ⚠️ 危险操作清单

| 函数                                                  | 风险行为                                 | 是否可逆 |
| ----------------------------------------------------- | ---------------------------------------- | -------- |
| `rm_files`                                            | 物理删除图片**及其同名标签**（`.jpg`/`.png`/`.jpeg` 同名都会带走），不进回收站；删除清单会逐条打印，一个没删到也会 `[warn]` 提示 | ❌       |
| `rm_dirs`                                             | 递归删除整个目录                         | ❌       |
| `process_files(mode=1)`                               | 逐个物理删除源文件                       | ❌       |
| `create_dirs(overwrite=True)`                         | 清空并重建已存在的目录                   | ❌       |
| `split_train_test(upset_photo=True)`                  | 清空目标目录中的现有文件后再划分；删除前逐条打印数量并等 `y` 确认，`assume_yes=True` 直接跳过确认 | ❌       |
| `split_meta`                                          | 删除并重建 `out_root`（默认 `./meta_split_results`），执行前打印 `[confirm]` 提示 | ❌       |
| `coco_to_txt` / `coco_txt_BaiDu` / `voc_to_txt`       | `save_dir` 已存在则**整个删除重建**      | ❌       |
| `rm_icc_profile`                                      | 就地覆盖写 PNG（单张损坏/只读不再中断整批，失败明细在返回的 `failures` 里） | ❌       |
| `normalize_labels`                                    | 就地覆盖写 txt 坐标；越界值双向钳位（`>1` 与 `<0` 都改），NaN 只上报不改动，明细在返回的 `nan_values` 里 | ❌       |
| `replace_labels_with_target` / `batch_replace_labels` | 就地覆盖写标签类别                       | ❌       |
| `add_suffix_to_files`                                 | 就地重命名文件                           | ❌       |
| `predict_high_dpi` / `save_with_dpi`                  | 输出目录中同名图片会被覆盖               | ❌       |

**建议**：动手前先备份，尤其是 `rm_*` 系列与 `upset_photo=True`。

---
