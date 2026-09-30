from ultralytics import YOLO, YOLOWorld


def evaluate(
    model_selection: str, yaml_data: str, yolo_world=False, iou: float = 0.7
) -> any:

    if yolo_world:
        result = YOLOWorld(model_selection).val(data=yaml_data, iou=iou)
    else:
        result = YOLO(model_selection).val(data=yaml_data, iou=iou, batch=1)

    return result
