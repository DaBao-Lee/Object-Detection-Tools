"""高 DPI 预测结果图生成。"""

import glob
import os
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ObjectDetection import get_best_last_model

# ---------------------------------------------------------------- 工具函数

# 模型权重
DEFAULT_WEIGHTS = (
    r"D:\A\硕士\代码方面\CICTP 2027\runs\detect\baseline\yolo11\weights\best.pt"
)
# 待预测图片 / 图片文件夹
DEFAULT_SOURCE = r"D:\A\硕士\代码方面\CICTP 2027\figures"
# 输出文件夹
DEFAULT_OUTPUT = r"D:\A\硕士\代码方面\CICTP 2027\outputs"

# Times New Roman 字族（Windows 自带）：regular / bold / italic / bold_italic
FONT_FILES = {
    "regular": r"C:\Windows\Fonts\times.ttf",
    "bold": r"C:\Windows\Fonts\timesbd.ttf",
    "italic": r"C:\Windows\Fonts\timesi.ttf",
    "bold_italic": r"C:\Windows\Fonts\timesbi.ttf",
}
DEFAULT_FONT_WEIGHT = "bold"

# 蓝色框（BGR，OpenCV 口径）
BOX_COLOR_BGR = (255, 0, 0)
# 标签文字白色（Pillow 口径，RGB）
TEXT_COLOR_RGB = (255, 255, 255)

IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp")


def collect_images(src: Union[str, list]) -> list:
    """把输入统一展开成图片路径列表。目录则递归收集其中的图片文件。"""

    if isinstance(src, list):
        return list(src)

    if os.path.isdir(src):
        imgs = []
        for ext in IMG_EXTS:
            imgs += glob.glob(os.path.join(src, f"**/*.{ext}"), recursive=True)
        return sorted(dict.fromkeys(imgs))

    return [src]


def resolve_font_path(font_weight: str) -> str:
    """把字重名（regular/bold/italic/bold_italic）解析成字体文件路径。

    也允许直接传一个字体文件路径。
    """

    if font_weight in FONT_FILES:
        return FONT_FILES[font_weight]
    return font_weight


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """载入字体，找不到时退回 Pillow 默认字体。"""

    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    print(f"[warn] 字体不存在：{font_path}，退回 Pillow 默认字体")
    return ImageFont.load_default(size=size)


def draw_detections(
    canvas_bgr: np.ndarray,
    detections: list,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    box_width: int = 3,
    box_color_bgr: tuple = BOX_COLOR_BGR,
    text_color_rgb: tuple = TEXT_COLOR_RGB,
    font_stroke: int = 0,
):
    """在 BGR 画布上画检测框与标签，返回新画布。

    detections: [(x1, y1, x2, y2, 类别名, 置信度), ...]

    box_width  蓝色框线宽（像素）
    font_stroke 字体描边宽度（0 = 不描边；>0 可在 Bold 基础上再加重）
    标签优先贴在框上方；顶到图像边缘则改贴框内顶部。标签底色与框同色，
    保证浅色背景下依然清晰。
    """

    # 1) 先用 OpenCV 画所有框（作用在 BGR 画布上）
    for x1, y1, x2, y2, _, _ in detections:
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), box_color_bgr, box_width)

    # 2) 再用 Pillow 在同一张图上写标签（RGB 口径）
    pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Pillow 需要 RGB，OpenCV 框色是 BGR，这里做一次转换
    box_color_rgb = tuple(int(c) for c in box_color_bgr[::-1])

    pad = max(2, int(font_size * 0.18))
    img_w, img_h = pil.size

    for x1, y1, x2, y2, cls_name, conf in detections:
        text = f"{cls_name} {conf:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=font_stroke)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 标签左上角：优先框上方，放不下落到框内顶部
        tx = x1
        ty = y1 - th - 2 * pad
        if ty < 0:
            ty = y1 + pad

        # 保证整个标签在画布内（左右都不越界）
        tx = max(0, min(tx, img_w - tw - 2 * pad))
        ty = max(0, min(ty, img_h - th - 2 * pad))

        draw.rectangle(
            [tx, ty, tx + tw + 2 * pad + 10, ty + th + 2 * pad + 10], fill=box_color_rgb
        )
        draw.text(
            (tx + pad, ty + pad),
            text,
            font=font,
            fill=text_color_rgb,
            stroke_width=font_stroke,
            stroke_fill=text_color_rgb,
        )

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def parse_size(reshape_size) -> tuple:
    """把 reshape_size 统一成 (宽, 高)；None 表示不缩放。"""

    if reshape_size is None:
        return None
    if isinstance(reshape_size, (tuple, list)):
        return int(reshape_size[0]), int(reshape_size[1])
    return int(reshape_size), int(reshape_size)


def save_with_dpi(image_bgr: np.ndarray, out_path: str, dpi: int = 300) -> str:
    """按指定 DPI 保存图片（PNG 无损，带 dpi 元数据）。"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if out_path.lower().endswith((".jpg", ".jpeg")):
        pil.save(out_path, format="JPEG", quality=95, dpi=(dpi, dpi))
    else:
        pil.save(out_path, format="PNG", dpi=(dpi, dpi))
    return out_path


# ---------------------------------------------------------------- 主函数


def predict_high_dpi(
    weights: str = DEFAULT_WEIGHTS,
    source: Union[str, list] = DEFAULT_SOURCE,
    output_dir: str = DEFAULT_OUTPUT,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = None,
    reshape_size: Union[int, tuple] = None,
    dpi: int = 300,
    # ---- 外观调节 ----
    font_weight: str = DEFAULT_FONT_WEIGHT,
    font_size: int = None,
    font_stroke: int = 0,
    box_width: int = None,
    # --------------------
    class_names: dict = None,
    scale: float = 1.0,
    save_format: str = "png",
    box_color_bgr: tuple = BOX_COLOR_BGR,
    text_color_rgb: tuple = TEXT_COLOR_RGB,
    show: bool = False,
):
    """出高 DPI 预测结果图。

    weights     模型权重路径
    source      单张图片路径 / 图片文件夹 / 图片路径列表
    output_dir  输出文件夹，默认 outputs
    conf        置信度阈值
    iou         NMS IoU 阈值
    imgsz       推理输入边长，默认 640；不受 reshape_size 影响
    reshape_size
                预测完成后把结果图缩放到该尺寸，框坐标同步换算。
                传 int 为正方形（如 320），传 (宽, 高) 可指定非正方形；None = 不缩放。
                框和标签是在缩放后的尺度上画的，所以文字不会被缩糊
    dpi         输出图片 DPI 元数据

    外观调节：
    font_weight 字重：regular / bold / italic / bold_italic，也可直接传字体文件路径
    font_size   字号（像素）；None = 按图幅自适应（短边 4.2%）
    font_stroke 字体描边宽度，0 = 不描边；>0 在字重基础上再加重
    box_width   蓝色框线宽（像素）；None = 按图幅自适应（短边 / 180）

    class_names 类别名映射，如 {0: 'crack'}；None 则用模型自带类别名
    scale       输出放大倍数，1.0 表示与输入分辨率一致
    save_format 输出格式，png / jpg
    """

    model = YOLO(weights)
    paths = collect_images(source)

    if not paths:
        print(f"[error] 没有找到图片：{source}")
        return []

    target = parse_size(reshape_size)
    # 推理仍在原图（imgsz 默认 640）上做，保证小目标检出率；reshape_size 只作用于输出
    eff_imgsz = imgsz or 640

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    font_path = resolve_font_path(font_weight)

    def render(result, src_path: str):
        """把单张结果画出来并按 dpi 存盘。"""

        canvas = result.orig_img.copy()
        h, w = canvas.shape[:2]

        # 输出尺寸：reshape_size 优先，再乘 scale
        tw, th = target if target else (w, h)
        tw = int(round(tw * scale))
        th = int(round(th * scale))

        # 先把底图缩放到输出尺寸，框坐标随后按同一比例换算，
        # 这样标签是直接画在输出尺度上的，文字不会被缩放糊掉
        if (tw, th) != (w, h):
            interp = cv2.INTER_AREA if tw * th < w * h else cv2.INTER_CUBIC
            canvas = cv2.resize(canvas, (tw, th), interpolation=interp)

        sx, sy = tw / w, th / h
        short = min(tw, th)

        # 字号 / 线宽：给了就用给的，否则按输出图幅自适应
        cur_font_size = font_size if font_size else max(16, int(short * 0.042))
        cur_box_width = box_width if box_width else max(2, int(round(short / 180)))

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, clss, confs):
                # 先裁回原图范围，再换算到输出尺度
                x1 = int(max(0, min(w, x1)) * sx)
                y1 = int(max(0, min(h, y1)) * sy)
                x2 = int(max(0, min(w, x2)) * sx)
                y2 = int(max(0, min(h, y2)) * sy)
                ci = int(c)
                name = (
                    class_names[ci]
                    if class_names is not None and ci in class_names
                    else result.names[ci]
                )
                detections.append((x1, y1, x2, y2, name, float(cf)))

        font = load_font(font_path, cur_font_size)
        canvas = draw_detections(
            canvas,
            detections,
            font,
            cur_font_size,
            box_width=cur_box_width,
            box_color_bgr=box_color_bgr,
            text_color_rgb=text_color_rgb,
            font_stroke=font_stroke,
        )

        stem = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(output_dir, f"{stem}.{save_format.lower()}")
        save_with_dpi(canvas, out_path, dpi=dpi)
        saved.append(out_path)

        print(
            f"[ok] {os.path.basename(src_path)} -> {out_path}  {tw}x{th}  目标 {len(detections)} 个"
        )

        if show:
            cv2.imshow("high-dpi predict", canvas)
            cv2.waitKey(0)

    # stream=True 逐张出结果，避免一次性把整批图读进内存
    results = model.predict(
        source=paths,
        conf=conf,
        iou=iou,
        imgsz=eff_imgsz,
        stream=True,
        verbose=False,
        save=False,
    )
    for src_path, result in zip(paths, results):
        render(result, src_path)

    if show:
        cv2.destroyAllWindows()

    if show:
        cv2.destroyAllWindows()

    print(f"[done] 共 {len(saved)} 张，输出目录：{output_dir}")
    return saved


# ---------------------------------------------------------------- 入口

if __name__ == "__main__":

    model_paths = [
        "./runs/detect/baseline/yolo5/weights/",
        "./runs/detect/baseline/yolo8/weights/",
        "./runs/detect/baseline/yolo9/weights/",
        "./runs/detect/baseline/yolo10/weights/",
        "./runs/detect/baseline/yolo11/weights/",
        "./runs/detect/baseline/yolo12/weights/",
        "./runs/detect/baseline/yolo26/weights/",
        "./runs/detect/ASGP-YOLO/ADown_SPPELAN_SE_56/weights/",
        "./runs/detect/ASGP-YOLO/ADown_SPPELAN_SE_56_GPC3k2/weights/",
    ]

    save_names = [
        "yolo5",
        "yolo8",
        "yolo9",
        "yolo10",
        "yolo11",
        "yolo12",
        "yolo26",
        "AS-YOLO",
        "ASGP-YOLO",
    ]

    # model_paths = [
    #     "./runs/detect/baseline/yolo11/weights/",
    #     "./runs/detect/ASGP-YOLO/ADown_SPPELAN_SE_56/weights/",
    #     "./runs/detect/ASGP-YOLO/ADown_SPPELAN_SE_56_GPC3k2/weights/",
    # ]

    # save_names = [
    #     "yolo11_test",
    #     "AS-YOLO_test",
    #     "ASGP-YOLO_test",
    # ]

    for model_path, save_name in zip(model_paths, save_names):
        predict_high_dpi(
            weights=get_best_last_model(model_path),
            source="./figures",
            output_dir=f"outputs/{save_name}",
            conf=0.4,
            iou=0.7,
            dpi=300,
            class_names={0: "crack"},
            reshape_size=320,
            font_size=25,
            box_width=4
        )
