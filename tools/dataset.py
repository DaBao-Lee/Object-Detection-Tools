"""数据集组织与划分：配对检查、train/val(/test) 切分、分块收集。

注意：本模块多个函数带删除 / 覆盖语义（rmtree、unlink），调用前确认路径并备份原始数据。
"""

import glob
import os
import random
import shutil
from typing import Union

from tqdm.auto import tqdm


def copy_files(
    src_files: Union[list, str],
    dest_dir: str,
    verbose: bool = True,
    max_num: int = None,
    assume_yes: bool = False,
) -> None:
    """复制文件到目标目录；同名文件会被跳过，可安全重跑。

    src_files 传目录时会自动展开为其中的文件；max_num 指定随机抽样的数量
    （默认交互确认，assume_yes=True 可跳过，供无头或批量调用）。
    verbose 为兼容旧调用保留，当前不产生输出。
    """

    if isinstance(src_files, str):
        src_files = glob.glob(os.path.join(src_files, "*.*"))
        if len(src_files) == 0:
            return

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    if max_num is not None:
        src_files = random.sample(src_files, min(max_num, len(src_files)))
        if not assume_yes:
            y_n = input(f"确定将{max_num}个目标文件复制到{dest_dir}?[y/n]")
            if y_n.lower() != "y":
                return

    for file in tqdm(src_files):
        dest_path = os.path.join(dest_dir, os.path.basename(file))
        if not os.path.exists(dest_path):
            shutil.copy2(file, dest_dir)


def check_and_copy_missing_files(
    img_path: str,
    label_path: str = None,
    label_type: str = "txt",
    meta_path: str = None,
    missing_dir: str = None,
) -> None:
    """检查图片 / 标签（可选元数据）的配对关系，找出有图无标签或有标签无图的样本。

    默认只打印文件名；给定 missing_dir 时把落单文件复制过去供人工复核。
    """

    imgs_name = [
        os.path.splitext(os.path.basename(x))[0]
        for x in glob.glob(os.path.join(img_path, "*.*"))
        if x.endswith((".jpg", ".png", "jpeg"))
    ]

    if label_path is not None:
        labels_name = [
            os.path.splitext(os.path.basename(x))[0]
            for x in glob.glob(os.path.join(label_path, f"*.{label_type}"))
        ]
        if len(imgs_name) >= len(labels_name):
            # 图片更多 → 找缺标签的图片
            for name in tqdm(imgs_name):
                if name not in labels_name:
                    print(f"{name} not in labels")
                    if missing_dir:
                        src_img = next(glob.glob(os.path.join(img_path, f"{name}.*")))
                        shutil.copy2(src_img, missing_dir)
        else:
            # 标签更多 → 找缺图片的标签
            for name in tqdm(labels_name):
                if name not in imgs_name:
                    print(f"{name} not in imgs")
                    if missing_dir:
                        src_label = next(
                            glob.glob(os.path.join(label_path, f"{name}.{label_type}"))
                        )
                        shutil.copy2(src_label, missing_dir)

    if meta_path is not None:
        meta_name = [
            os.path.splitext(os.path.basename(x))[0]
            for x in glob.glob(os.path.join(meta_path, "*.*"))
            if x.endswith((".jpg", ".png", "jpeg"))
        ]
        for name in tqdm(meta_name):
            if name not in imgs_name:
                print(f"{name} not in imgs")
                if missing_dir:
                    src_meta = next(glob.glob(os.path.join(meta_path, f"{name}.*")))
                    shutil.copy2(src_meta, missing_dir)


def rm_files(src_files: str, target_file: str, label_type: str = "txt") -> None:
    """删除指定文件及其同名标签（图和标签要一起删，否则破坏配对）。

    注意：直接物理删除，不进回收站。
    """

    assert os.path.exists(src_files), f"{src_files} 不存在"
    files = glob.glob(os.path.join(src_files, "*.*"))

    target_file_base = os.path.splitext(target_file)[0]
    target_txt = os.path.join(src_files, target_file_base + f".{label_type}")
    target_img = [
        os.path.join(src_files, target_file_base + ext)
        for ext in [".jpg", ".png", ".jpeg"]
    ]

    if target_txt in files:
        os.remove(target_txt)

    for img in target_img:
        if img in files:
            os.remove(img)


def split_meta(
    meta_path: str,
    split_name: str = "meta",
    split_num: int = 5,
    match: bool = True,
    label_type: str = "txt",
) -> None:
    """把目录里的图片和标签平均切成 split_num 份，输出到 ./meta_split_results/{split_name}{i}/。

    match=True 时先取图片与标签的交集再划分。注意：输出目录是相对当前工作目录的，
    且 ./meta_split_results 已存在会被整体删除重建。
    """

    imgs = []
    for ext in ["jpg", "png", "jpeg"]:
        imgs.extend(glob.glob(os.path.join(meta_path, f"*.{ext}")))
    labels = glob.glob(os.path.join(meta_path, f"*.{label_type}"))

    if match:
        # 以文件名主干为键求交集，保证图、标签成对
        img_dict = {os.path.splitext(os.path.basename(img))[0]: img for img in imgs}
        label_dict = {
            os.path.splitext(os.path.basename(label))[0]: label for label in labels
        }
        common_keys = set(img_dict.keys()).intersection(set(label_dict.keys()))
        imgs = [img_dict[key] for key in common_keys]
        labels = [label_dict[key] for key in common_keys]

    if len(imgs) != len(labels):
        return  # 图、标签数量不一致，放弃本次划分

    length = len(imgs) // split_num + 1
    if os.path.exists("./meta_split_results"):
        shutil.rmtree("./meta_split_results")
    os.mkdir("./meta_split_results")
    for i in range(split_num):
        os.mkdir(f"./meta_split_results/{split_name}{i}")
        for index in range(i * length, length * (i + 1)):
            try:
                # 最后一份通常会下标越界，直接跳过
                shutil.copy2(imgs[index], f"./meta_split_results/{split_name}{i}")
                shutil.copy2(labels[index], f"./meta_split_results/{split_name}{i}")
            except:
                pass


def collcet_meta(
    meta_split_path: str = "./meta_split_results", save: bool = False
) -> list:
    """收集 split_meta 各子目录下的所有文件路径；save=True 时另复制一份到 ./meta。"""

    if not os.path.exists(meta_split_path):
        return  # 没有划分结果，直接返回

    split_results = glob.glob(os.path.join(meta_split_path, "*"))
    meta = []
    for split_result in split_results:
        meta.extend(glob.glob(os.path.join(split_result, "*.*")))

    if save:
        os.mkdir("./meta")
        copy_files(meta, "./meta", verbose=False)

    return meta


def spilt_train_test(
    li_path: Union[str, list],
    train_img_path: str,
    val_img_path: str,
    train_label_path: str,
    val_label_path: str,
    label_type: str = "txt",
    negative_path: str = None,
    val_size: Union[int, float] = 0.2,
    random_state: int = 110,
    upset_photo: bool = False,
    verbose=True,
    need_test: bool = False,
):
    """划分训练 / 验证（可选测试）集并复制落盘。

    关键参数：val_size 为比例(float)或数量(int)；negative_path 指定负样本目录（混入训练集压制误检）；
    upset_photo=True 会先清空目标目录再划分；need_test=True 需同时设 upset_photo=True。

    注意：need_test 分支固定从训练集抽 100 张作测试集，样本不足会报错；
    upset_photo=True 会物理删除目标目录中的现有文件。
    """

    path = li_path
    
    if not os.path.exists(train_img_path): os.makedirs(train_img_path)
    if not os.path.exists(val_img_path): os.makedirs(val_img_path)
    if not os.path.exists(train_label_path): os.makedirs(train_label_path)
    if not os.path.exists(val_label_path): os.makedirs(val_label_path)
    
    random.seed(random_state)  # 固定种子，保证划分可复现

    if isinstance(li_path, str):
        img_path = (
            glob.glob(os.path.join(path, "*.jpg"))
            + glob.glob(os.path.join(path, "*.png"))
            + glob.glob(os.path.join(path, "*.jpeg"))
        )
        label_path = [
            x
            for x in glob.glob(os.path.join(path, f"*.{label_type}"))
            if "class" not in x
        ]

    if isinstance(li_path, list):
        img_path = [x for x in li_path if x.endswith((".jpg", ".png", ".jpeg"))]
        label_path = [x for x in li_path if x.endswith(f".{label_type}")]

    if isinstance(val_size, float):
        val_img = random.sample(img_path, int(len(img_path) * val_size))
    else:
        val_img = random.sample(img_path, val_size)

    val_label_names = [os.path.splitext(os.path.basename(x))[0] for x in val_img]
    val_label = [
        x
        for x in label_path
        if os.path.splitext(os.path.basename(x))[0] in val_label_names
    ]
    train_img = [x for x in img_path if x not in val_img]

    train_label = [x for x in label_path if x not in val_label]

    if upset_photo:
        t_img = train_img_path
        v_img = val_img_path

        t_label = train_label_path
        v_label = val_label_path

        for item in [t_img, v_img, t_label, v_label]:
            if not os.path.exists(item):
                os.makedirs(item)

        tree = (
            glob.glob(os.path.join(t_img, "*.*"))
            + glob.glob(os.path.join(v_img, "*.*"))
            + glob.glob(os.path.join(t_label, "*.*"))
            + glob.glob(os.path.join(v_label, "*.*"))
        )

        if need_test:
            te_img = train_img_path.replace("train", "test")
            te_label = train_label_path.replace("train", "test")

            if not os.path.exists(te_img):
                os.makedirs(te_img)
            if not os.path.exists(te_label):
                os.makedirs(te_label)

            test_img = random.sample(train_img, 100)
            test_name = [os.path.splitext(os.path.basename(x))[0] for x in test_img]
            test_label = [
                x
                for x in label_path
                if os.path.splitext(os.path.basename(x))[0] in test_name
            ]

            # 测试集必须从训练集中移除，否则同一张图会同时出现在 train / test
            for img in test_img:
                train_img.remove(img)

            for label in test_label:
                train_label.remove(label)

            tree += glob.glob(os.path.join(te_img, "*.*")) + glob.glob(
                os.path.join(te_label, "*.*")
            )

        for file in tree:
            os.unlink(file)

    # 负样本随机混入训练集，上限 2500 张且不超过训练集规模的 10%
    if negative_path is not None:
        train_img += random.sample(
            glob.glob(os.path.join(negative_path, "*.jpg")),
            min(2500, int(len(train_img) * 0.1)),
        )

    copy_files(train_img, train_img_path, verbose)
    copy_files(val_img, val_img_path, verbose)
    copy_files(train_label, train_label_path, verbose)
    copy_files(val_label, val_label_path, verbose)

    if need_test:
        copy_files(test_img, te_img, verbose)
        copy_files(test_label, te_label, verbose)
