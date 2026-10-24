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

    🔴 rect 是最容易踩的评测陷阱，同一个权重两种口径能差约 **0.040**：
        rect=True  → YOLO11n 实测 mAP50-95 = 0.7061（= 训练日志口径）
        rect=False → YOLO11n 实测 mAP50-95 = 0.6656
      官方文档：rect=True 按宽高分组、每维向上取整到 stride(32) 倍数（640→672），
        且**正方形图像 rect=False 与 predict 结果一致**。若图片 100% 正方形，
        走 model.predict() 的自建评测必然低报约 0.04。

    📌 两套口径并存，**不可混用**：
        · **同一框架内多模型横向对比 → 用 rect=True**（本函数当前默认），
          与训练日志内部口径统一，横向可比。
        · **跨框架对比（如 YOLO vs NanoDet）→ 必须 rect=False**。
          NanoDet 侧 keep_ratio=False，没有 rect 机制；用 True 比会让 YOLO 白占 0.04。

    ⚠️ 跨表引用数字时务必标口径，例如「0.7061 (rect=True)」。

    ⚠️ rect=False 不可用「输入改成 672」替代（实测 GPC3k2：F640=0.6523 → F672=0.6519，
       几乎无变化，离 rect=True 的 0.7083 仍差 0.056）。resize 补不回这个差。

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
