"""标注格式转换：COCO / 百度 COCO / VOC → YOLO txt。
YOLO 每行格式为 ``cls x_center y_center w h``，后 4 位归一化到 0~1。
"""

import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np
from lxml import etree
from tqdm.auto import tqdm
from ultralytics.data.converter import convert_coco


def coco_to_txt(annotations_path: str, save_dir: str, use_segments=True) -> None:
    """COCO json → YOLO txt（官方转换器，类别按 cls91to80 映射）。

    save_dir 已存在会被删除重建；use_segments=True 用分割标注，否则用检测框。
    """

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    convert_coco(annotations_path, save_dir=save_dir, use_segments=use_segments, lvis=False, cls91to80=True)


def coco_txt_BaiDu(annotations_path: str, save_dir: str) -> None:
    """百度风格 COCO json → YOLO txt（自解析，不做类别映射，跳过 iscrowd 标注）。

    annotations_path 传目录；save_dir 已存在会被删除重建。
    """

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for file in os.listdir(annotations_path):
        js = json.load(open(os.path.join(annotations_path, file)))
        images = {f'{x["id"]:d}': x for x in js["images"]}

        # 按 image_id 分组，避免每张图重复扫描整个标注列表
        imgToAnns = defaultdict(list)
        for ann in js["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        for img_id, anns in tqdm(imgToAnns.items()):
            img = images[f"{img_id:d}"]
            f = img["file_name"]
            h, w = img["height"], img["width"]
            bboxes = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # COCO bbox = [x_min, y_min, w, h] → 中心点 + 按宽高归一化
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = ann["category_id"]
                bboxes.append([cls] + box.tolist())
            with open(os.path.join(save_dir, str(f).split('.')[0] + '.txt'), '+a') as t:
                for bbox in bboxes:
                    t.write("%g " % bbox[0])
                    for i in bbox[1:]:
                        t.write("%g " % i)
                    t.write("\n")


def voc_to_txt(annotations_path: str, save_dir: str, obj_dict: dict = None) -> None:
    """VOC xml → YOLO txt。

    save_dir 已存在会被删除重建；obj_dict 为类别名→数字 id 的映射
    （必填，例如 {"wheel": 0, "handle": 1}），未覆盖的类别名会抛 KeyError。
    """

    assert obj_dict is not None, '请输入对象名称与类别编号的字典'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    xmls = glob.glob(os.path.join(annotations_path, '*.xml'))
    for xml_name in tqdm(xmls):
        txt_name = os.path.basename(xml_name).replace('xml', 'txt')
        f = open(os.path.join(save_dir, txt_name), '+w')
        with open(xml_name, 'rb') as fp:
            xml = etree.HTML(fp.read())
            width = int(xml.xpath('//size/width/text()')[0])
            height = int(xml.xpath('//size/height/text()')[0])

            obj = xml.xpath('//object')
            for each in obj:
                name = each.xpath("./name/text()")[0]
                classes = obj_dict[name]
                xmin = int(each.xpath('./bndbox/xmin/text()')[0])
                xmax = int(each.xpath('./bndbox/xmax/text()')[0])
                ymin = int(each.xpath('./bndbox/ymin/text()')[0])
                ymax = int(each.xpath('./bndbox/ymax/text()')[0])

                # VOC 是绝对像素的左上 / 右下角，转中心点并归一化
                dw = 1 / width
                dh = 1 / height
                x_center = (xmin + xmax) / 2
                y_center = (ymax + ymin) / 2
                w = (xmax - xmin)
                h = (ymax - ymin)
                x, y, w, h = x_center * dw, y_center * dh, w * dw, h * dh

                f.write(str(classes) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + '\n')
        f.close()
