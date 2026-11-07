"""ObjectDetection —— 检测工具包

convert 标注转换 · preprocess 数据清洗与标签治理 · dataset 数据集划分 · train 训练 ·
val 验证与模型复杂度统计 · predict 推理（批量图片 + .pt / .onnx 单帧流式双引擎） ·
draw_results 检测结果渲染（画框、日志面板、窗口居中） · export 导出 ·
model_utils 权重定位 · systools 通用文件操作 ·
video_process 视频抽帧与内存取帧 · open_api 视觉大模型接口调用。

`high_dpi_predict` 未并入本文件，需从子模块导入（它用的是包内相对导入，
只能在仓库的父目录下按包模式跑）：
    from ObjectDetection.high_dpi_predict import predict_high_dpi

注意：部分函数带删除 / 覆盖 / 重命名语义，调用前确认路径并备份原始数据。
"""

from .convert import coco_txt_BaiDu, coco_to_txt, voc_to_txt
from .dataset import (
    check_and_copy_missing_files,
    collect_meta,
    copy_files,
    rm_files,
    split_meta,
    split_train_test,
)
from .draw_results import (
    BOX_COLOR,
    DEFAULT_FONT,
    FONT_FILES,
    PANEL_ALPHA,
    PANEL_COLOR,
    PANEL_FONT,
    TASKBAR_OFFSET,
    TEXT_COLOR,
    Detection,
    PredictResult,
    center_window,
    draw_boxes,
    draw_log_panel,
    finish_predict,
    get_screen_size,
    load_font,
)
from .export import export_model
from .model_utils import get_best_last_model, list_weights
from .open_api import DEFAULT_SYSTEM_PROMPT, OpenAPI, chat_stream, getSystemPrompt
from .predict import (
    InferDataType,
    add_pillow_text,
    get_infer_data,
    letterbox,
    nms_per_class,
    onnx_predict,
    predict,
    pt_predict,
    warmup_pt,
)
from .preprocess import (
    IMAGE_EXTS,
    batch_replace_labels,
    collect_images,
    count_labels,
    fill_empty_labels,
    normalize_labels,
    replace_labels_with_target,
    rm_icc_profile,
)
from .systools import (
    add_suffix_to_files,
    create_dirs,
    fetch_specific_files,
    ignore_something,
    process_files,
    rm_dirs,
)
from .train import train
from .val import calc_complexity_yolo, evaluate
from .video_process import (
    VIDEO_EXTS,
    extract_frames,
    iter_frames,
    read_frame,
    video_info,
)

__all__ = [
    # 训练 / 验证 / 复杂度
    "train",
    "evaluate",
    "calc_complexity_yolo",
    # 批量图片推理
    "predict",
    "add_pillow_text",
    "InferDataType",
    "get_infer_data",
    # 单帧流式推理 · .pt 引擎
    "pt_predict",
    "warmup_pt",
    # 单帧流式推理 · .onnx 引擎
    "onnx_predict",
    "letterbox",
    "nms_per_class",
    # 导出与权重定位
    "export_model",
    "get_best_last_model",
    "list_weights",
    # 检测结果渲染
    "Detection",
    "PredictResult",
    "finish_predict",
    "draw_boxes",
    "draw_log_panel",
    "load_font",
    "get_screen_size",
    "center_window",
    "FONT_FILES",
    "DEFAULT_FONT",
    "BOX_COLOR",
    "TEXT_COLOR",
    "PANEL_FONT",
    "PANEL_COLOR",
    "PANEL_ALPHA",
    "TASKBAR_OFFSET",
    # 数据处理
    "coco_to_txt",
    "coco_txt_BaiDu",
    "voc_to_txt",
    "rm_icc_profile",
    "normalize_labels",
    "copy_files",
    "check_and_copy_missing_files",
    "rm_files",
    "split_meta",
    "collect_meta",
    "split_train_test",
    # 标签治理
    "IMAGE_EXTS",
    "replace_labels_with_target",
    "batch_replace_labels",
    "collect_images",
    "fill_empty_labels",
    "count_labels",
    # 通用文件操作
    "create_dirs",
    "rm_dirs",
    "fetch_specific_files",
    "process_files",
    "ignore_something",
    "add_suffix_to_files",
    # 视频抽帧与内存取帧
    "extract_frames",
    "VIDEO_EXTS",
    "video_info",
    "iter_frames",
    "read_frame",
    # 视觉大模型接口调用
    "OpenAPI",
    "chat_stream",
    "getSystemPrompt",
    "DEFAULT_SYSTEM_PROMPT",
]
