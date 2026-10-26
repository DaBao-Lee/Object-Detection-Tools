"""推理预测。

predict 用 ultralytics 内置绘图；advanced_predict 自绘框，支持中文标签。
"""

import glob
import os
import random
from enum import Enum
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, YOLOWorld


class InferDataType(Enum):
    """推理数据类型：IMAGE 随机抽样若干张，DIR 取整个目录。"""

    IMAGE = "image"
    DIR = "dir"


# 推理可接受的图片扩展名（大小写不敏感）
IMAGE_EXTS = (".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _collect_images(directory: str, recursive: bool = False) -> list:
    """收集目录下的图片路径，按扩展名后缀过滤，可选递归子目录。"""

    pattern = (
        os.path.join(directory, "**", "*")
        if recursive
        else os.path.join(directory, "*")
    )
    return sorted(
        path
        for path in glob.glob(pattern, recursive=recursive)
        if os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )


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
    yolo_world=False,
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

    if yolo_world:
        model = YOLOWorld(model_selection)
    else:
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


def advanced_predict(
    model_selection: str,
    img_path: Union[str, list],
    yolo_world=False,
    conf: float = 0.5,
    save: bool = False,
    show: bool = True,
    font_path=None,
    replace_text: dict = None,
    iou: float = 0.7,
):
    """自绘检测框与标签的预测。

    font_path 给中文字体路径即可显示中文；replace_text 按 ID 覆盖类别名（未给出的沿用模型名称），
    例如 {0: '裂缝'}。save=True 时存到 ./results/（相对当前工作目录）；show=True 时每张图都会阻塞等待按键。
    """

    if yolo_world:
        model = YOLOWorld(model_selection)
    else:
        model = YOLO(model_selection)

    # 传目录时展开为其中的所有图片；单文件（任意扩展名 / 大小写）原样交给模型
    if isinstance(img_path, str) and os.path.isdir(img_path):
        img_path = _collect_images(img_path)

    results = model(img_path, conf=conf, show=False, save=False, verbose=False, iou=iou)
    for result in results:
        names = result.names
        if replace_text:
            names = {**names, **replace_text}
        boxes = result.boxes
        orig_img = result.orig_img
        path = result.path

        for index in range(len(boxes)):
            cls_ = names[boxes[index].cls.cpu().item()]
            x1, y1, x2, y2 = np.int_(boxes.xyxy[index].cpu().numpy())
            # 坐标裁剪回图像范围内，避免框超出画布
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)

            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (220, 110, 10), 3)

            text_x = max(0, x1)
            text_y = y1

            # 铺一块实心色块作文字背景，浅色图上也能看清
            text_bg_color = (220, 110, 10)
            cv2.rectangle(
                orig_img,
                (text_x, text_y - 1),
                (text_x + 50, text_y + 23),
                text_bg_color,
                -1,
            )
            if font_path is not None:
                orig_img = add_pillow_text(
                    orig_img, font_path, cls_, (text_x, text_y), textSize=22
                )
            else:
                cv2.putText(
                    orig_img,
                    cls_,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        if show:
            cv2.imshow("result", orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(os.path.join("results", os.path.basename(path)), orig_img)
