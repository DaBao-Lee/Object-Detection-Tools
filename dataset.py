"""数据集组织与划分：配对检查、train/val(/test) 切分、分块收集。

函数清单：
    copy_files                    复制文件到目标目录，同名跳过
    check_and_copy_missing_files  检查图片 / 标签 / 元数据的配对关系
    rm_files                      删除文件及其同名标签
    split_meta                    把目录平均切成若干份
    collect_meta                  收拢 split_meta 的各份结果
    split_train_test              划分训练 / 验证（可选测试）集

注意：本模块多个函数带删除 / 覆盖语义（rmtree、unlink、upset_photo），调用前确认路径并备份原始数据。
"""

import glob
import os
import random
import shutil
from typing import Union

from tqdm.auto import tqdm

# 数据集侧认的图片扩展名。比 preprocess.IMAGE_EXTS 窄是有意的：
# 划分时多算一种格式等于换了一份训练数据，要扩就先把已有数据集重跑一遍再改这里
IMG_EXTS = (".jpg", ".png", ".jpeg")


def _is_img(path: str) -> bool:
    """按扩展名判断是不是图片（大小写不敏感，且必须带点——早期写法漏了点，notesjpeg 也算图片）。"""

    return os.path.splitext(path)[1].lower() in IMG_EXTS


def _stem(path: str) -> str:
    """文件名主干，配对比较一律以它为准（图 a.jpg 与标签 a.txt 算同一对）。"""

    return os.path.splitext(os.path.basename(path))[0]


def _sample(pool: list, num: int, what: str) -> list:
    """随机抽样，数量不合法时给一句人话，而不是让 random 抛 'too many samples'。"""

    if not 0 <= num <= len(pool):
        raise ValueError(
            f"{what} 要抽 {num} 个，候选只有 {len(pool)} 个，请调整数量或先补齐数据"
        )

    return random.sample(pool, num)


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

    if max_num is not None:
        src_files = _sample(list(src_files), min(max_num, len(src_files)), "复制源文件")
        if not assume_yes:
            y_n = input(f"确定将{len(src_files)}个目标文件复制到{dest_dir}?[y/n]")
            if y_n.lower() != "y":
                return

    os.makedirs(dest_dir, exist_ok=True)

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

    默认只打印文件名；给定 missing_dir 时把落单文件复制过去供人工复核（目录不存在会新建）。
    """

    imgs = [x for x in glob.glob(os.path.join(img_path, "*")) if _is_img(x)]
    imgs_name = {_stem(x) for x in imgs}

    if missing_dir:
        os.makedirs(missing_dir, exist_ok=True)

    def _rescue(name: str, directory: str, pattern: str) -> None:
        """把落单文件复制去复核；找不到同名文件时说明清楚，而不是抛一句看不懂的 StopIteration。"""

        if not missing_dir:
            return
        hits = glob.glob(os.path.join(directory, pattern.format(name=name)))
        if not hits:
            print(f"[warn] {name} 在 {directory} 里找不到对应文件，跳过复制")
            return
        shutil.copy2(hits[0], missing_dir)

    if label_path is not None:
        labels = glob.glob(os.path.join(label_path, f"*.{label_type}"))
        labels_name = {_stem(x) for x in labels}

        # 谁多就以谁为准找落单方；数量相等时两边都查一遍，免得「一样多但配不上」被漏掉
        if len(imgs) >= len(labels):
            for name in tqdm(sorted(imgs_name)):
                if name not in labels_name:
                    print(f"{name} not in labels")
                    _rescue(name, img_path, "{name}.*")
        else:
            for name in tqdm(sorted(labels_name)):
                if name not in imgs_name:
                    print(f"{name} not in imgs")
                    _rescue(name, label_path, "{name}." + label_type)

    if meta_path is not None:
        metas = [x for x in glob.glob(os.path.join(meta_path, "*")) if _is_img(x)]
        for name in tqdm(sorted(_stem(x) for x in metas)):
            if name not in imgs_name:
                print(f"{name} not in imgs")
                _rescue(name, meta_path, "{name}.*")


def rm_files(src_files: str, target_file: str, label_type: str = "txt") -> list:
    """删除指定文件及其同名标签（图和标签要一起删，否则破坏配对）。

    target_file 传文件名或带目录的路径都行，只按文件名主干匹配。
    注意：直接物理删除，不进回收站。返回删掉的路径列表。
    """

    assert os.path.isdir(src_files), f"{src_files} 不存在或不是目录"

    stem = _stem(target_file)
    candidates = [os.path.join(src_files, f"{stem}.{label_type}")] + [
        os.path.join(src_files, stem + ext) for ext in IMG_EXTS
    ]

    removed = []
    for path in candidates:
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)

    if removed:
        print(f"已删除 {len(removed)} 个文件：{', '.join(os.path.basename(p) for p in removed)}")
    else:
        print(f"[warn] {src_files} 里没有主干为 {stem} 的文件或标签，什么都没删")

    return removed


def _even_chunks(total: int, num: int) -> list:
    """把 0..total-1 平均切成 num 段，余数摊给前几段；num 大于 total 时缩到 total 段，不留空段。"""

    num = min(num, total)
    base, extra = divmod(total, num)

    chunks = []
    start = 0
    for i in range(num):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size

    return chunks


def split_meta(
    meta_path: str,
    split_name: str = "meta",
    split_num: int = 5,
    match: bool = True,
    label_type: str = "txt",
    out_root: str = "./meta_split_results",
) -> list:
    """把目录里的图片和标签平均切成 split_num 份，输出到 {out_root}/{split_name}{i}/。

    match=True 时先取图片与标签的交集再划分。
    注意：out_root 默认是相对当前工作目录的；已存在会被整体删除重建（旧版会切出一个空目录，
    是因为它用 len//num+1 分段再靠吞下标越界兜住，现在改成真正平均分配）。
    返回各份的文件路径列表。
    """

    imgs = [x for x in glob.glob(os.path.join(meta_path, "*")) if _is_img(x)]
    labels = glob.glob(os.path.join(meta_path, f"*.{label_type}"))

    if match:
        # 以文件名主干为键求交集，保证图、标签成对
        img_dict = {_stem(img): img for img in imgs}
        label_dict = {_stem(label): label for label in labels}
        common = sorted(set(img_dict) & set(label_dict))
        imgs = [img_dict[k] for k in common]
        labels = [label_dict[k] for k in common]

    if len(imgs) != len(labels):
        print(f"[warn] 图片 {len(imgs)} 张 / 标签 {len(labels)} 个 数量不一致，放弃本次划分")
        return []

    if split_num < 1:
        raise ValueError(f"split_num 至少为 1，收到 {split_num}")

    if os.path.exists(out_root):
        doomed = glob.glob(os.path.join(out_root, "*"))
        print(f"[confirm] 将删除并重建 {out_root}（现有 {len(doomed)} 项）")
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    groups = []
    for i, index in enumerate(_even_chunks(len(imgs), split_num)):
        dst = os.path.join(out_root, f"{split_name}{i}")
        os.makedirs(dst, exist_ok=True)
        files = []
        for k in tqdm(index, desc=f"{split_name}{i}", unit="对", leave=False):
            shutil.copy2(imgs[k], dst)
            shutil.copy2(labels[k], dst)
            files += [
                os.path.join(dst, os.path.basename(imgs[k])),
                os.path.join(dst, os.path.basename(labels[k])),
            ]
        groups.append(files)

    print(f"[done] {len(imgs)} 对图标签切成 {len(groups)} 份 → {out_root}")
    return groups


def collect_meta(
    meta_split_path: str = "./meta_split_results", save: bool = False
) -> list:
    """收集 split_meta 各子目录下的所有文件路径；save=True 时另复制一份到 ./meta。"""

    if not os.path.exists(meta_split_path):
        print(f"[warn] 没有划分结果：{meta_split_path} 不存在")
        return []

    meta = []
    for split_result in glob.glob(os.path.join(meta_split_path, "*")):
        meta.extend(glob.glob(os.path.join(split_result, "*.*")))

    if save:
        copy_files(meta, "./meta", verbose=False)

    return meta


def _to_test_dir(train_dir: str, arg_name: str) -> str:
    """由训练目录推出测试目录：只换最后一级目录名里的 train。

    旧版对整个路径做 str.replace，D:\\training_data\\train\\img 会连父目录一起被改坏。
    """

    parent, base = os.path.split(os.path.normpath(train_dir))
    if "train" not in base:
        raise ValueError(
            f"推不出测试目录：{train_dir} 的最后一级目录名里没有 train，"
            f"请显式传 {arg_name}"
        )

    return os.path.join(parent, base.replace("train", "test"))


def split_train_test(
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
    verbose: bool = True,
    need_test: bool = False,
    test_img_path: str = None,
    test_label_path: str = None,
    test_size: int = 100,
    assume_yes: bool = False,
) -> None:
    """划分训练 / 验证（可选测试）集并复制落盘。

    关键参数：val_size 为比例(float)或数量(int)；negative_path 指定负样本目录（混入训练集压制误检）；
    upset_photo=True 会先清空目标目录再划分；need_test=True 必须同时 upset_photo=True
    （测试集是从训练集里挑出来的，不清空就没法保证 train / test 不重样）。

    测试目录默认由 train_img_path / train_label_path 的最后一级目录名推出来，
    推不出来（目录名里没有 train）时报错，此时显式传 test_img_path / test_label_path。
    test_size 想改默认 100 张的口子也在这里。
    """

    if need_test and not upset_photo:
        raise ValueError(
            "need_test=True 需要同时设 upset_photo=True："
            "测试集从训练集里抽，不清空目标目录的话 train/test 会留下重复样本"
        )

    for item in [train_img_path, val_img_path, train_label_path, val_label_path]:
        os.makedirs(item, exist_ok=True)

    random.seed(random_state)  # 固定种子，保证划分可复现

    if isinstance(li_path, str):
        img_path = [x for x in glob.glob(os.path.join(li_path, "*")) if _is_img(x)]
        # classes.txt 这类类别表跟标签同扩展名，按文件名主干排除（旧版拿整条路径做子串匹配，
        # 目录名里带 class 会把整个文件夹的标签全滤掉）
        label_path = [
            x
            for x in glob.glob(os.path.join(li_path, f"*.{label_type}"))
            if not _stem(x).lower().startswith("class")
        ]
    elif isinstance(li_path, list):
        img_path = [x for x in li_path if _is_img(x)]
        label_path = [x for x in li_path if str(x).lower().endswith(f".{label_type}")]
    else:
        raise TypeError(f"li_path 只能是目录或路径列表，收到 {type(li_path).__name__}")

    if not img_path:
        raise ValueError(
            f"{li_path} 下没找到图片；当前只认 {IMG_EXTS}，"
            f"要收别的格式请先把它们转成这几种（口径见模块开头 IMG_EXTS 的说明）"
        )

    n_val = int(len(img_path) * val_size) if isinstance(val_size, float) else int(val_size)
    val_img = _sample(img_path, n_val, f"验证集（共 {len(img_path)} 张图）")

    val_names = {_stem(x) for x in val_img}
    val_label = [x for x in label_path if _stem(x) in val_names]
    val_set = set(val_img)
    train_img = [x for x in img_path if x not in val_set]
    train_label = [x for x in label_path if _stem(x) not in val_names]

    if need_test:
        te_img = test_img_path or _to_test_dir(train_img_path, "test_img_path")
        te_label = test_label_path or _to_test_dir(train_label_path, "test_label_path")

    if upset_photo:
        dirs = [train_img_path, val_img_path, train_label_path, val_label_path]
        if need_test:
            dirs += [te_img, te_label]

        tree = [f for d in dirs for f in glob.glob(os.path.join(d, "*"))]

        # 测试集必须从训练集里挑，且挑完要从训练集移除，否则同一张图同时出现在 train / test
        if need_test:
            n_test = min(test_size, len(train_img))
            if n_test < test_size:
                print(
                    f"[warn] 测试集想要 {test_size} 张，训练集只剩 {len(train_img)} 张，"
                    f"按 {n_test} 张切"
                )
            test_img = random.sample(train_img, n_test)
            test_names = {_stem(x) for x in test_img}
            test_label = [x for x in label_path if _stem(x) in test_names]

            test_set = set(test_img)
            train_img = [x for x in train_img if x not in test_set]
            train_label = [x for x in train_label if _stem(x) not in test_names]

        if tree:
            print(f"[confirm] upset_photo=True：将物理删除 {len(tree)} 个现有文件")
            if not assume_yes:
                y_n = input("确定删除?[y/n]")
                if y_n.lower() != "y":
                    print("已取消，什么都没动")
                    return
            for file in tree:
                os.unlink(file)

        for d in dirs:
            os.makedirs(d, exist_ok=True)

    # 负样本随机混入训练集，上限 2500 张且不超过训练集规模的 10%
    if negative_path is not None:
        negatives = [x for x in glob.glob(os.path.join(negative_path, "*")) if _is_img(x)]
        train_img += random.sample(
            negatives, min(2500, int(len(train_img) * 0.1), len(negatives))
        )

    copy_files(train_img, train_img_path, verbose)
    copy_files(val_img, val_img_path, verbose)
    copy_files(train_label, train_label_path, verbose)
    copy_files(val_label, val_label_path, verbose)

    if need_test:
        copy_files(test_img, te_img, verbose)
        copy_files(test_label, te_label, verbose)
