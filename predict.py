"""推理预测：批量图片推理（predict）与单帧流式推理（.pt / .onnx 双引擎）。

predict 走 ultralytics 内置绘图；两个单帧引擎只负责出框，画框、打面板、弹窗、组装
PredictResult 全在 draw_results.finish_predict，出图口径两引擎一致。

onnxruntime 只有跑 .onnx 引擎时才需要：本文件用守卫导入，没装也能 import ObjectDetection，
只在调用 onnx_predict 时才明确报错。
"""

from __future__ import annotations

import ast
import os
import random
import time
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np

# torch 必须排在 onnxruntime 前面：靠 torch 把 cublas/cudnn 所在目录注册进 DLL 搜索路径，
# CUDA provider 才找得到它们；单独 os.add_dll_directory 不管用。这两处顺序别调换
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .draw_results import (
    BOX_COLOR,
    TEXT_COLOR,
    Detection,
    PredictResult,
    finish_predict,
)
from .preprocess import collect_images
from .video_process import iter_frames


class InferDataType(Enum):
    """推理数据类型：IMAGE 随机抽样若干张，DIR 取整个目录。"""

    IMAGE = "image"
    DIR = "dir"


# 推理可接受的图片扩展名（大小写不敏感）。比 preprocess.IMAGE_EXTS 窄是有意的：
# 推理目录里常混着 gif / heic 这类解码口径不明的图，宁可这里不收
IMAGE_EXTS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _collect_images(directory: str, recursive: bool = False) -> list:
    """收集目录下的图片路径，按推理侧 IMAGE_EXTS 过滤，可选递归子目录。"""

    imgs = collect_images(Path(directory), recursive, set(IMAGE_EXTS))
    return [str(p) for p in imgs]


def get_infer_data(
    path_dir: str,
    typing: InferDataType = InferDataType.IMAGE,
    max_num: int = 5,
    recursive: bool = False,
):
    """按类型取推理数据，返回图片路径列表（IMAGE 为随机抽样，上限 max_num）。

    按扩展名后缀判断图片（大小写不敏感）；recursive=True 时递归搜索子目录。
    """

    imgs = _collect_images(path_dir, recursive=recursive)

    if typing == InferDataType.IMAGE:
        return random.sample(imgs, min(max_num, len(imgs)))

    return imgs


def predict(
    model_selection: str,
    img_path: str,
    conf: float = 0.5,
    save: bool = False,
    show: bool = True,
    verbose: bool = True,
    stream: bool = False,
    iou: float = 0.7,
):
    """用模型预测，返回 ultralytics 的推理结果。

    show=True 时逐张弹窗显示检测结果、按任意键切换，批量跑图请置 False；
    save=True 时由 ultralytics 存到 runs/detect/predict*/。
    """

    model = YOLO(model_selection)

    results = model.predict(
        source=img_path,
        conf=conf,
        show=False,
        save=save,
        verbose=verbose,
        stream=stream,
        iou=iou,
        line_width=1,
    )

    # show 判断放在循环外：show=False 时不遍历，避免 stream=True 的生成器被提前耗光
    if show:
        for result in results:
            cv2.imshow("result", result.plot())
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def add_pillow_text(
    img_array: np.array,
    font_path: str,
    text: str,
    position,
    textColor=(255, 255, 255),
    textSize: int = 20,
):
    """在图像数组（BGR）上写文本，返回新数组。

    OpenCV 的 putText 不支持中文，这里走 Pillow 的 truetype 字体。
    """

    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ==================== 单帧流式推理 · .pt 引擎 ====================
# 预处理与 NMS 全部由 ultralytics 内部完成，这一层只做「取缓存模型 → predict → 归一化成 Detection」

# YOLO() 一次加载约 1s，逐帧调用会吃掉整帧预算，所以按权重绝对路径复用同一个实例
_PT_MODEL_CACHE: Dict[str, YOLO] = {}


# 用途：按路径取（或首次加载并缓存）YOLO 实例；手里已有实例的调用方直接透传，不再建新的
def _load_pt_model(weights: Union[str, os.PathLike, YOLO]) -> YOLO:
    if isinstance(weights, YOLO):
        return weights

    path = os.fspath(weights)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"权重文件不存在：{path}")

    # 绝对路径做 key：同一个权重写成 "./weights/a.pt" 和 "weights\\a.pt" 只该占一份
    key = os.path.abspath(path)
    if key not in _PT_MODEL_CACHE:
        _PT_MODEL_CACHE[key] = YOLO(key)
        # 先入缓存再预热：否则 warmup_pt 里的 _load_pt_model 会再次进来，无限递归
        warmup_pt(_PT_MODEL_CACHE[key])
    return _PT_MODEL_CACHE[key]


# 用途：用一张全黑假帧走一遍 predict，把 CUDA 初始化、kernel 首次编译这些一次性开销提前掉。
#       加载时自动调一次（实测 1 次就到收益上限：首帧 3441 ms → 218 ms，再多压不会更快）；
#       想手动再压一遍也可以直接调这个函数
def warmup_pt(weights: Union[str, os.PathLike, YOLO], size: int = 640) -> None:
    model = _load_pt_model(weights)
    # 假帧只为走通和真实帧完全相同的 predict 路径，输出直接丢，不画框不存盘不显示
    model.predict(source=np.zeros((size, size, 3), dtype=np.uint8), verbose=False)


# 用途：.pt 单帧检测入口。参数分四组——推理（conf/iou/imgsz/device）、
#       画图（box_*/label_bg/font*）、面板（draw/show/show_panel/panel_font_size）、
#       显示（display_size）
def pt_predict(
    weights: Union[str, os.PathLike, YOLO],
    frame: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: Optional[str] = None,
    box_color: Tuple[int, int, int] = BOX_COLOR,
    box_width: Optional[int] = None,
    label_bg: Optional[Tuple[int, int, int]] = None,
    text_color: Tuple[int, int, int] = TEXT_COLOR,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    font_stroke: int = 0,
    class_names: Optional[Dict[int, str]] = None,
    draw: bool = True,
    show: bool = False,
    show_panel: bool = True,
    panel_font_size: Optional[int] = None,
    # display_size 传 (宽, 高)，只影响 result.display 和弹窗，annotated 仍是原尺寸
    display_size: Optional[Tuple[int, int]] = None,
) -> PredictResult:

    model = _load_pt_model(weights)

    # 返回给调用方的 orig 是副本：画框发生在另一张图上，调用方手里的帧不能被改动
    orig = frame.copy()
    h, w = orig.shape[:2]

    # 耗时走 perf_counter、时刻走 time.time：系统对时会把 wall clock 往回拨，
    # 用两个 time.time 相减可能得到负耗时；perf_counter 单调但没有日历意义
    t0 = time.perf_counter()
    start_ts = time.time()

    # source 传单帧时返回长度为 1 的列表，直接取第 0 个
    result = model.predict(
        source=orig,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    # 类名优先用调用方传的，其次用权重自带的 metadata，都没有就退成数字 id 字符串
    names = class_names or result.names or {}
    detections: List[Detection] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # 这三个属性是 torch 张量、可能还在 GPU 上，.cpu().numpy() 之后才能当普通数组用
        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cid, cf in zip(xyxy_arr, cls_arr, conf_arr):
            cid = int(cid)
            detections.append(
                Detection(
                    cls_id=cid,
                    cls_name=names.get(cid, str(cid)),
                    conf=float(cf),
                    # letterbox 还原后坐标是浮点且边缘能溢出 1~2 px，
                    # 钳进图幅再取整，免得 cv2 画到画布外、下游拿到越界或负坐标
                    xyxy=(
                        int(max(0, min(w - 1, x1))),
                        int(max(0, min(h - 1, y1))),
                        int(max(0, min(w - 1, x2))),
                        int(max(0, min(h - 1, y2))),
                    ),
                )
            )

    return finish_predict(
        orig,
        detections,
        infer_ms,
        t0,
        start_ts,
        box_color=box_color,
        box_width=box_width,
        label_bg=label_bg,
        text_color=text_color,
        font=font,
        font_size=font_size,
        font_stroke=font_stroke,
        draw=draw,
        show=show,
        show_panel=show_panel,
        panel_font_size=panel_font_size,
        window="pt_predict",
        display_size=display_size,
    )


# ==================== 单帧流式推理 · .onnx 引擎 ====================
# ultralytics 导出 onnx 时把后处理剪掉了，所以这一层要自己做前处理、解码和 NMS

_ONNX_SESSIONS: Dict[Tuple[str, Tuple[str, ...]], "ort.InferenceSession"] = {}


# 用途：探测本机能力，决定默认跑在 GPU 还是 CPU
def _default_providers() -> List[str]:
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# 用途：按「路径 + providers」取（或首次建）InferenceSession；已建好的 session 直接透传
def _load_onnx_model(
    weights: Union[str, os.PathLike, "ort.InferenceSession"],
    providers: Optional[Sequence[str]] = None,
) -> "ort.InferenceSession":
    if ort is None:
        raise ImportError(
            "未安装 onnxruntime，.onnx 引擎不可用："
            "装 onnxruntime-gpu（配 CUDA）或 onnxruntime（纯 CPU）后再调 onnx_predict"
        )

    if isinstance(weights, ort.InferenceSession):
        return weights

    path = os.fspath(weights)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"权重文件不存在：{path}")

    key = os.path.abspath(path)
    # providers 必须一起进 key：只按路径缓存的话，第二次显式传 CPU 会拿到先建好的 CUDA
    # session，参数静默失效，测出来的耗时是假的
    cache_key = (key, tuple(providers or _default_providers()))
    if cache_key not in _ONNX_SESSIONS:
        _ONNX_SESSIONS[cache_key] = ort.InferenceSession(
            key, providers=list(cache_key[1])
        )
    return _ONNX_SESSIONS[cache_key]


# 用途：从 session 元数据里还原类别名；导出时 ultralytics 把 names 写成了 dict 字面量字符串
def _onnx_names(session: "ort.InferenceSession") -> Dict[int, str]:
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if not raw:
        return {}
    # 元数据只有字符串，用 ast 而不是 eval：解析字面量不会执行任意代码；
    # 键是 "0"/"1" 这种，转回 int 才能和 cls id 对上
    try:
        return {int(k): str(v) for k, v in ast.literal_eval(raw).items()}
    except (ValueError, SyntaxError):
        return {}


# 用途：等比缩放后居中贴进 size×size 的 114 灰画布，并返回反算坐标要用的 scale 和 pad
def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    # 114 是 ultralytics 训练时的填充灰，换成别的值等于给模型看它没见过的边框
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    # pad 取整跟 ultralytics 的 Letterbox 对齐（-0.1 再 round），反算坐标时两边才咬得上
    left = int(round((size - nw) / 2 - 0.1))
    top = int(round((size - nh) / 2 - 0.1))
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, left, top


# 用途：逐类 NMS，返回保留框在原候选数组里的下标。
#       按类分组是为了跨类不互相压制——免得裂缝框把挨着的坑洞框筛掉
def nms_per_class(
    xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thr: float
) -> List[int]:

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep: List[int] = []

    for cid in np.unique(cls_ids):
        # 同一类内部按分数从高到低排队，先处理的框优先保留
        cand = np.nonzero(cls_ids == cid)[0]
        cand = cand[np.argsort(-scores[cand])]
        while len(cand) > 0:
            # 队首当锚框收进 keep，然后拿它和其余框算 IoU，超阈值的（同一目标的重复检出的）丢掉
            head, rest = cand[0], cand[1:]
            keep.append(int(head))
            if len(rest) == 0:
                break
            # 交并比向量化：一次对 rest 全部算交集宽高，不写 Python 循环
            inter = np.maximum(
                0,
                np.minimum(xyxy[head, 2], xyxy[rest, 2])
                - np.maximum(xyxy[head, 0], xyxy[rest, 0]),
            ) * np.maximum(
                0,
                np.minimum(xyxy[head, 3], xyxy[rest, 3])
                - np.maximum(xyxy[head, 1], xyxy[rest, 1]),
            )
            union = areas[head] + areas[rest] - inter
            # union 可能为 0（退化成点的框），where 先挡住除零，按 IoU=0 处理成不重叠
            cand = rest[np.where(union > 0, inter / union, 0.0) <= iou_thr]

    return keep


# 用途：把网络原始输出解析成钳进图幅的 Detection 列表（转置判别 → 阈值筛 → 坐标反算 → NMS）
def _onnx_postprocess(
    raw: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    img_w: int,
    img_h: int,
    conf: float,
    iou: float,
    names: Dict[int, str],
) -> List[Detection]:

    out = np.squeeze(raw)
    # ultralytics 导出多为 [1, 4+nc, anchors]（特征在行），也有 [1, anchors, 4+nc]；
    # 特征数恒小于候选数，比一下大小就知道要不要转置
    if out.shape[0] < out.shape[1]:
        out = out.T

    # 转置完每行是一个候选框：前 4 列 cx cy w h，其余每列一个类别的概率
    scores = out[:, 4:]
    # 一行里哪一列概率最高就说明是那一类，那个概率就是这个框的置信度
    cls_ids = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)

    # 先按阈值粗筛再进 NMS：8400 个候选全丢进逐类 NMS 会慢一个数量级
    sel = confs >= conf
    if not np.any(sel):
        return []

    boxes, cls_ids, confs = out[sel, :4], cls_ids[sel], confs[sel]

    # 框是 640 画布上的中心点格式：先减 pad 再除 scale 才回到原帧像素
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack(
        [
            (cx - bw / 2 - pad_x) / scale,
            (cy - bh / 2 - pad_y) / scale,
            (cx + bw / 2 - pad_x) / scale,
            (cy + bh / 2 - pad_y) / scale,
        ],
        axis=1,
    )

    keep = nms_per_class(xyxy, confs, cls_ids, iou)

    detections: List[Detection] = []
    for i in keep:
        cid = int(cls_ids[i])
        detections.append(
            Detection(
                cls_id=cid,
                cls_name=names.get(cid, str(cid)),
                conf=float(confs[i]),
                xyxy=(
                    int(max(0, min(img_w - 1, xyxy[i, 0]))),
                    int(max(0, min(img_h - 1, xyxy[i, 1]))),
                    int(max(0, min(img_w - 1, xyxy[i, 2]))),
                    int(max(0, min(img_h - 1, xyxy[i, 3]))),
                ),
            )
        )
    return detections


# 用途：.onnx 单帧检测入口，参数和 pt_predict 一一对应，只把 device/imgsz 换成
#       providers（imgsz 由输入节点 shape 自己决定）
def onnx_predict(
    weights: Union[str, os.PathLike, "ort.InferenceSession"],
    frame: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.7,
    providers: Optional[Sequence[str]] = None,
    box_color: Tuple[int, int, int] = BOX_COLOR,
    box_width: Optional[int] = None,
    label_bg: Optional[Tuple[int, int, int]] = None,
    text_color: Tuple[int, int, int] = TEXT_COLOR,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    font_stroke: int = 0,
    class_names: Optional[Dict[int, str]] = None,
    draw: bool = True,
    show: bool = False,
    show_panel: bool = True,
    panel_font_size: Optional[int] = None,
    display_size: Optional[Tuple[int, int]] = None,
) -> PredictResult:

    session = _load_onnx_model(weights, providers)
    orig = frame.copy()
    h, w = orig.shape[:2]

    input_node = session.get_inputs()[0]
    # 动态轴导出时 shape[2] 是字符串占位，拿不到具体边长就退回 640
    size = input_node.shape[2] if isinstance(input_node.shape[2], int) else 640

    # 计时口径和 pt_predict 一致（perf_counter 算时长、time.time 记时刻）；
    # 这里 infer_ms 从 letterbox 起表，前处理本来就是 onnx 推理管线的一部分，拆不开
    t0 = time.perf_counter()
    start_ts = time.time()

    padded, scale, pad_x, pad_y = letterbox(orig, size)
    # BGR→RGB、HWC→CHW、除 255：这批 ONNX 是 ultralytics dynamic=False 导的，归一化没固化进图，
    # 不除 255 会让分数饱和到 1.0、候选从几个暴涨到几百个
    blob = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    )
    # transpose/ascontiguousarray 之后仍是 3×size×size，[None, ...] 补上 batch 维；
    # 第一个参数给 None 表示取全部输出节点
    raw = session.run(None, {input_node.name: blob[None, ...]})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    detections = _onnx_postprocess(
        raw, scale, pad_x, pad_y, w, h, conf, iou, class_names or _onnx_names(session)
    )

    return finish_predict(
        orig,
        detections,
        infer_ms,
        t0,
        start_ts,
        box_color=box_color,
        box_width=box_width,
        label_bg=label_bg,
        text_color=text_color,
        font=font,
        font_size=font_size,
        font_stroke=font_stroke,
        draw=draw,
        show=show,
        show_panel=show_panel,
        panel_font_size=panel_font_size,
        window="onnx_predict",
        display_size=display_size,
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":

    # 示例视频与权重：换成你自己的路径即可
    SOURCE = r"D:\A\硕士\代码方面\巡检小车\stream_predict\videos\2.mp4"
    PT_WEIGHTS = r"D:\A\硕士\代码方面\巡检小车\stream_predict\weights\yolov8n.pt"
    ONNX_WEIGHTS = r"D:\A\硕士\代码方面\巡检小车\stream_predict\weights\yolov8n.onnx"

    # .pt 引擎：模型按绝对路径缓存，只在第一帧真正加载
    for idx, frame in iter_frames(SOURCE):
        r = pt_predict(weights=PT_WEIGHTS, frame=frame, conf=0.5, imgsz=640)
        cv2.imshow("frame", r.annotated)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    # .onnx 引擎：出图口径与 .pt 完全一致，只是换了推理后端
    if ort is not None:
        for idx, frame in iter_frames(SOURCE):
            r = onnx_predict(weights=ONNX_WEIGHTS, frame=frame, conf=0.5)
            cv2.imshow("frame", r.annotated)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
