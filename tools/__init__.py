"""tools —— 无人车巡检 / 桥梁裂缝检测 工程工具包。

convert 标注转换 · preprocess 数据清洗与标签治理 · dataset 数据集划分 · train 训练 ·
val 验证 · predict 推理 · export 导出 · model_utils 权重定位 ·
compare_metrics 训练指标对比可视化 · systools 通用文件操作 · video 视频抽帧。

注意：部分函数带删除 / 覆盖 / 重命名语义，调用前确认路径并备份原始数据。
"""

from .convert import coco_txt_BaiDu, coco_to_txt, voc_to_txt
from .compare_metrics import (
    METRICS,
    Metric,
    compare_metrics,
    expand_csv_inputs,
    last_n_stats,
    list_metrics,
    load_results_csv,
    plot_curves,
    read_run_label,
    resolve_metrics,
    smooth_series,
)
from .dataset import (
    check_and_copy_missing_files,
    collcet_meta,
    copy_files,
    rm_files,
    spilt_train_test,
    split_meta,
)
from .export import export_model
from .model_utils import get_best_last_model
from .predict import (
    InferDataType,
    add_pillow_text,
    advanced_predict,
    get_infer_data,
    predict,
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
from .val import evaluate
from .video import extract_frames

__all__ = [
    # 训练 / 验证 / 推理 / 导出
    "train",
    "evaluate",
    "predict",
    "advanced_predict",
    "add_pillow_text",
    "export_model",
    "get_best_last_model",
    "InferDataType",
    "get_infer_data",
    # 训练指标对比可视化
    "compare_metrics",
    "list_metrics",
    "resolve_metrics",
    "load_results_csv",
    "expand_csv_inputs",
    "read_run_label",
    "smooth_series",
    "last_n_stats",
    "plot_curves",
    "METRICS",
    "Metric",
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
    "collcet_meta",
    "spilt_train_test",
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
    # 视频抽帧
    "extract_frames",
]
