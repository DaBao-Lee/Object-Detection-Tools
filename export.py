"""模型导出：把 .pt 权重导出为 ONNX / TensorRT / OpenVINO 等部署格式。"""

from ultralytics import YOLO


def export_model(model_selection: str, format: str = "onnx"):
    """导出模型，产物与权重同目录、同名不同后缀。

    format 常用值：onnx / torchscript / openvino / engine（TensorRT，需匹配 CUDA）。
    """

    YOLO(model_selection).export(format=format)
