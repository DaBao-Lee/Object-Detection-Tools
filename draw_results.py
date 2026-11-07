"""检测结果渲染层：拿到 List[Detection] 之后的一切（画框、打面板、弹窗、组装 PredictResult）都收在这里，
引擎只负责出框，出图口径各引擎完全一致。
"""

from __future__ import annotations

import os
import time

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 渲染默认值与结果类型 ====================
# 各条链路产出同一批类型、共用同一批渲染默认值，所以只在这里定义一次。


# font 参数既可以是这里的键名，也可以直接是 ttf 绝对路径（走 load_font 解析）
FONT_FILES = {
    "regular": r"C:\Windows\Fonts\times.ttf",
    "bold": r"C:\Windows\Fonts\timesbd.ttf",
    "italic": r"C:\Windows\Fonts\timesi.ttf",
    "bold_italic": r"C:\Windows\Fonts\timesbi.ttf",
}
DEFAULT_FONT = "bold"

# 颜色统一按 BGR 收，和 OpenCV 帧口径一致；需要 Pillow 时内部再翻成 RGB
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

# 面板要等宽（比例字体下 "count" 和 "start_time" 两列对不齐），且小字号必须加粗才不发虚
PANEL_FONT = r"C:\Windows\Fonts\consolab.ttf"
PANEL_COLOR = (40, 40, 40)
PANEL_ALPHA = 0.55


# 用途：单个检测框的规范结构；各条链路都归一到这个格式，下游不必区分引擎
@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


# 用途：一帧的完整产出——原图、标注图、显示尺寸图、框列表、计数、两种耗时、起止时刻
# 可变字段（list / dict）必须用 field(default_factory=...)，直接写 = [] 定义类时就报错
@dataclass
class PredictResult:
    # 送进来那一帧的副本：框和面板都画在别的图上，调用方手里的原帧不会被改动
    orig: np.ndarray
    annotated: Optional[np.ndarray]  # 原尺寸的成品图；draw=False 且不打面板时为 None
    # 显示尺寸版本：annotated 缩放后的图，None = 没给 display_size 或目标尺寸和原图相同
    display: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)  # 坐标已取整并钳进图幅
    count: int = 0  # 本帧框数，恒等于 len(detections)
    # 类名 -> 该类框数；面板中间那几行（类别有几种就几行）就是它
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    # 引擎自己那段，两条链路口径不同：pt 含 ultralytics 内部的预处理+推理+NMS；
    # onnx 含 letterbox+归一化+session.run，但不含它自己那步 Python 解码/NMS
    infer_ms: float = 0.0
    # 整帧耗时：从引擎起点到 finish_predict 里的结算点，含画框和标签，不含画面板与弹窗
    elapsed_ms: float = 0.0
    # time.time() 墙钟秒，只用来记「什么时候」；算时长走 perf_counter，
    start_ts: float = 0.0
    end_ts: float = 0.0
    # 上面两个的 ISO 串（精确到毫秒），面板最后两行 start_time / end_time 就是它
    start_time: str = ""
    end_time: str = ""


# ---------- 字体 ----------

# 缺字体只警告一次的记号牌，进程内累加，重启即清空
_FONT_WARNED = set()


# 用途：把字体名（FONT_FILES 的键）或 ttf 路径解析成 Pillow 字体对象，缺文件退默认字体
def load_font(font: Optional[str], size: int):
    path = FONT_FILES.get(font or DEFAULT_FONT, font)
    if path and os.path.isfile(path):
        return ImageFont.truetype(path, size)

    # 缺字体只警告一次：视频逐帧调用时每次都 print 会把控制台刷满
    if font not in _FONT_WARNED:
        _FONT_WARNED.add(font)
        print(f"[warn] 字体文件不存在：{path}，改用 Pillow 默认字体")
    return ImageFont.load_default(size=size)


# ---------- 画框 + 标签 ----------


# 用途：在帧副本上画矩形框和「类别 置信度」标签，输出仍是 BGR
def draw_boxes(
    canvas: np.ndarray,
    detections: List[Detection],
    font,
    font_size: int,
    box_width: int,
    box_color: Tuple[int, int, int],
    label_bg: Optional[Tuple[int, int, int]],
    text_color: Tuple[int, int, int],
    font_stroke: int = 0,
) -> np.ndarray:

    # 两段式画法：直线矩形用 cv2（快、颜色按 BGR 直接给），
    # 标签要字号/字重/描边这些排印能力，cv2 做不了，翻成 RGB 交给 Pillow
    out = canvas.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, box_width)

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Pillow 吃 RGB，入参统一按 BGR 收，这里翻通道；int() 是给 numpy 标量兜底
    bg_rgb = tuple(int(c) for c in (label_bg or box_color)[::-1])
    fg_rgb = tuple(int(c) for c in text_color[::-1])
    pad = max(2, int(font_size * 0.18))
    img_w, img_h = pil.size

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        text = f"{det.cls_name} {det.conf:.2f}"
        # textbbox 量的是「文字占多大」，用来决定底色块尺寸和落点，不能写死
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=font_stroke)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 标签默认挂在框上方；框贴着画幅上沿就改挂框内侧，最后再钳进图幅
        tx = x1
        ty = y1 - th - 2 * pad
        if ty < 0:
            ty = y1 + pad

        tx = max(0, min(tx, img_w - tw - 2 * pad))
        ty = max(0, min(ty, img_h - th - 2 * pad))

        draw.rectangle(
            [tx, ty, tx + tw + 2 * pad + 10, ty + th + 2 * pad + 10], fill=bg_rgb
        )
        draw.text(
            (tx + pad, ty + pad),
            text,
            font=font,
            fill=fg_rgb,
            stroke_width=font_stroke,
            stroke_fill=fg_rgb,
        )

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ---------- 日志面板 ----------


# 用途：左上角叠一块半透明底，逐行写等宽对齐的「键 值」日志
def draw_log_panel(
    canvas: np.ndarray,
    lines: Sequence[str],
    font_size: int,
    bg_color: Tuple[int, int, int] = PANEL_COLOR,
    alpha: float = PANEL_ALPHA,
    text_color: Tuple[int, int, int] = TEXT_COLOR,
) -> np.ndarray:

    out = canvas.copy()
    img_h, img_w = out.shape[:2]
    pad = max(3, int(font_size * 0.4))
    line_h = int(font_size * 1.35)

    font = load_font(PANEL_FONT, font_size)
    # 面板宽度得按最长那行算。拿 1×1 的临时画布只为借它的 textlength 量字宽，不需要真图
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    panel_w = int(max(probe.textlength(text, font=font) for text in lines)) + 2 * pad
    panel_h = line_h * len(lines) + 2 * pad

    # 小图 + 长键名时面板可能比画面还大，钳一下左上角，别把画布撑出负坐标
    x0 = max(0, min(pad, img_w - panel_w))
    y0 = max(0, min(pad, img_h - panel_h))

    # 半透明底：cv2 没有 alpha 概念，只能用 addWeighted 对 ROI 做一次混合
    roi = out[y0 : y0 + panel_h, x0 : x0 + panel_w]
    out[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(
        roi, 1 - alpha, np.full_like(roi, bg_color), alpha, 0
    )

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fg_rgb = tuple(int(c) for c in text_color[::-1])
    for i, text in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * line_h), text, font=font, fill=fg_rgb)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# ---------- 窗口居中 ----------

# 底部任务栏高度(px)：居中时整体上抬半个任务栏，免得画面下沿被压住
TASKBAR_OFFSET = 44

# 已经居过的 (窗口名, 宽, 高)：逐帧调用也只真搬一次，之后用户自己拖窗口不会被下一帧拽回去
_CENTERED: set = set()


# 用途：读屏幕的物理分辨率 (宽, 高)，给 center_window 当基准
def get_screen_size() -> Tuple[int, int]:

    if os.name == "nt":
        try:
            import ctypes  # 局部导入，非 Windows 平台用不到

            user32 = ctypes.windll.user32
            # 先声明 DPI 感知，否则 150% 缩放屏会拿到 1280x720 这种逻辑像素假值
            user32.SetProcessDPIAware()
            w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return int(w), int(h)
        except Exception:
            pass

    return 1920, 1080


# 用途：把窗口摆到屏幕正中，返回实际设的外框左上角；搬不动（headless 等）给 (-1, -1)
def center_window(
    window: str,
    shape: Tuple[int, int],
    taskbar_offset: int = TASKBAR_OFFSET,
) -> Tuple[int, int]:

    h, w = shape[0], shape[1]
    screen_w, screen_h = get_screen_size()

    # 先算客户区（画面本身）左上角该落在哪：屏幕中心减去半幅画面
    cx = screen_w // 2 - w // 2
    cy = (screen_h - taskbar_offset) // 2 - h // 2

    # moveWindow 吃的是「窗口外框」左上角，比客户区多一圈装饰边，得倒扣掉。
    # 本机实测（Win 26300 + OpenCV 5.0 + 125% 缩放）：左边框 9px、标题栏 38px
    x = max(cx - (9 if os.name == "nt" else 2), 0)
    y = max(cy - (38 if os.name == "nt" else 24), 0)

    try:
        # OpenCV 5.0 的 moveWindow 只收 (name, x, y)，窗口尺寸由图像本身决定；
        # 窗口必须已经 imshow 过，否则 moveWindow 静默无效
        cv2.moveWindow(window, x, y)
        return x, y
    except Exception:
        return -1, -1


# ---------- 统一收口 ----------


# 用途：「已经拿到 Detection 列表」之后的统计、画图、面板、缩放、弹窗、组装结果全在这做完，
#       pt_predict 和 onnx_predict 各自只管出框，尾巴共享这一条
def finish_predict(
    orig: np.ndarray,
    detections: List[Detection],
    infer_ms: float,
    t0: float,
    start_ts: float,
    *,
    box_color: Tuple[int, int, int],
    box_width: Optional[int],
    label_bg: Optional[Tuple[int, int, int]],
    text_color: Tuple[int, int, int],
    font: Optional[str],
    font_size: Optional[int],
    font_stroke: int,
    draw: bool,
    show: bool,
    show_panel: bool,
    panel_font_size: Optional[int],
    window: str,
    display_size: Optional[Tuple[int, int]],
) -> PredictResult:

    h, w = orig.shape[:2]

    counts_by_class: Dict[str, int] = {}
    for det in detections:
        counts_by_class[det.cls_name] = counts_by_class.get(det.cls_name, 0) + 1

    # 自适应基准是输出图幅短边：标签和框线沿用 high_dpi_predict.py 的系数，
    # 面板字号单独定（12px 时白字占比只有粗体的一半，笔画太细看着发虚）
    short = min(w, h)
    cur_font_size = font_size if font_size else max(16, int(short * 0.042))
    cur_box_width = box_width if box_width else max(2, int(round(short / 180)))
    cur_panel_font = panel_font_size if panel_font_size else max(14, int(short * 0.036))

    annotated: Optional[np.ndarray] = None
    if draw:
        annotated = draw_boxes(
            orig,
            detections,
            load_font(font, cur_font_size),
            cur_font_size,
            cur_box_width,
            box_color,
            label_bg,
            text_color,
            font_stroke,
        )
    elif show_panel:
        # 不画框但要打面板，就拿干净帧当底板
        annotated = orig.copy()

    # 时间在这里结算：面板要显示这两个数，所以画面板和弹窗自身不计进 elapsed_ms
    end_ts = time.time()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    start_str = datetime.fromtimestamp(start_ts).isoformat(
        sep=" ", timespec="milliseconds"
    )
    end_str = datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="milliseconds")

    if show_panel:
        # 行序：总数 → 各类别（类别有几种就几行） → 推理耗时 → 起止时刻
        rows = [
            ("count", len(detections)),
            *counts_by_class.items(),
            ("infer_ms", f"{infer_ms:.3f}"),
            ("start_time", start_str),
            ("end_time", end_str),
        ]
        # 等宽字体 + 定宽键名，两列才能对齐
        annotated = draw_log_panel(
            annotated, [f"{key:<12}{val}" for key, val in rows], cur_panel_font
        )

    # 显示口径：框和面板全画完之后整体缩放到 display_size(w, h)，
    # 缩的是成品图，所以 Detection.xyxy 仍然是原图坐标，不跟着换算
    display: Optional[np.ndarray] = None
    if annotated is not None and display_size is not None:
        w_out, h_out = int(display_size[0]), int(display_size[1])
        if (w_out, h_out) != (w, h):
            # 缩小用 INTER_AREA（不会毛刺），放大用 INTER_CUBIC（边缘不糊）
            interp = cv2.INTER_AREA if w_out * h_out < w * h else cv2.INTER_CUBIC
            display = cv2.resize(annotated, (w_out, h_out), interpolation=interp)

    if show:
        # 弹窗优先用显示尺寸那份；没给 display_size（或尺寸等于原图）时就用原尺寸的 annotated
        preview = display if display is not None else annotated
        shown = preview if preview is not None else orig
        cv2.imshow(window, shown)
        # 非阻塞：视频逐帧调用时 waitKey(0) 会把每一帧都停在等按键
        cv2.waitKey(1)
        # 居中：同一个窗口同一幅尺寸只搬一次，换个 display_size 尺寸变了会重新居中
        key = (window, shown.shape[1], shown.shape[0])
        if key not in _CENTERED:
            _CENTERED.add(key)
            center_window(window, shown.shape)

    return PredictResult(
        orig=orig,
        annotated=annotated,
        display=display,
        detections=detections,
        count=len(detections),
        counts_by_class=counts_by_class,
        infer_ms=round(infer_ms, 3),
        elapsed_ms=round(elapsed_ms, 3),
        start_ts=start_ts,
        end_ts=end_ts,
        start_time=start_str,
        end_time=end_str,
    )
