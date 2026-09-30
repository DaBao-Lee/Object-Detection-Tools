"""模型导出：把 .pt 权重导出为 ONNX / TensorRT / OpenVINO 等部署格式。"""

from ultralytics import YOLO, YOLOWorld


def export_model(model_selection: str, yolo_world: bool = False, format: str = "onnx"):
    """导出模型，产物与权重同目录、同名不同后缀。

    format 常用值：onnx / torchscript / openvino / engine（TensorRT，需匹配 CUDA）。
    """

    if yolo_world:
        model = YOLOWorld(model_selection)
    else:
        model = YOLO(model_selection)

    model.export(format=format)
