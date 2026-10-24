from ultralytics import YOLO, YOLOWorld


def evaluate(
    model_selection: str,
    yaml_data: str,
    yolo_world: bool = False,
    iou: float = 0.7,
    batch: int = 1,
    rect: bool = True,
    imgsz: int = 640,
    conf: float = 0.001,
) -> any:
    """YOLO 验证，返回 ultralytics 的验证结果对象。

    model_selection : 权重路径（.pt）
    yaml_data       : 数据集 yaml 路径
    yolo_world      : True 走 YOLOWorld，False 走 YOLO
    iou             : **NMS 的 IoU 阈值**（默认 0.7），不是 mAP 的判定阈值，别混淆
    batch           : 验证批大小
    rect            : 是否按宽高分组推理（见上方说明）
    imgsz           : 输入尺寸，默认 640
    conf            : 置信度阈值，保持 0.001（ultralytics BaseValidator 默认值）。
                    调高会截断 PR 曲线，导致 P/R 和 mAP 失真。
    """

    if yolo_world:
        result = YOLOWorld(model_selection).val(
            data=yaml_data, iou=iou, batch=batch, rect=rect, imgsz=imgsz, conf=conf
        )
    else:
        result = YOLO(model_selection).val(
            data=yaml_data, iou=iou, batch=batch, rect=rect, imgsz=imgsz, conf=conf
        )

    return result
