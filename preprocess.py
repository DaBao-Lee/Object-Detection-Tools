"""图像 / 标签预处理：去 PNG 的 ICC 色彩配置、夹紧越界坐标、标签类别统一、空标签补全、类别计数。

函数清单：
    rm_icc_profile              删除目录内所有 PNG 的 ICC 色彩配置
    normalize_labels            把标签坐标夹紧到 (0, 1) 以内
    replace_labels_with_target  把标签文件里指定类别的 ID 改成目标类别
    batch_replace_labels        整个文件夹的类别 ID 批量替换
    count_labels                统计标签文件里各类别的标注数量（txt / xml）
    collect_images              收集目录下的图片文件
    fill_empty_labels           为缺失标签的图片生成空标签文件
"""

import glob
import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

# count_labels 默认扫描的标签类型；XML 按 VOC 约定解析
LABEL_EXTS = (".txt", ".xml")
XML_OBJECT_TAG = "object"  # 每个标注框所在的标签名
XML_NAME_TAGS = ("name", "label")  # 类别名所在子标签，按顺序取第一个非空值

# 常见图片扩展名集合，供 collect_images / fill_empty_labels 复用。
# 注意 dataset.IMG_EXTS 是更窄的一套：划分数据集时多算一种格式等于换训练数据，两边不要对齐
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".jfif",
    ".pjpeg",
    ".avif",
    ".heic",
}


def rm_icc_profile(src_path: str) -> dict:
    """删除目录内所有 PNG 的 ICC 色彩配置。注意：就地覆盖写，不可逆。

    先把像素 load() 干净再截断重写——就地保存会打开同一个文件写入，
    Pillow 的解码是惰性的，不先 load 就是在读自己正在截断的那份数据。
    单张图失败只记进 failures，不再中断整批。
    返回 {"total", "done", "failures"}。
    """

    paths = glob.glob(os.path.join(src_path, "*.png"))
    done = 0
    failures: list[tuple[str, str]] = []

    for path in tqdm(paths):
        try:
            with Image.open(path) as img:
                img.load()
                img.save(path, format="PNG", icc_profile=None)
            done += 1
        except Exception as exc:
            failures.append((path, str(exc)))
            tqdm.write(f"❌ {os.path.basename(path)} 去 ICC 失败: {exc}")

    if failures:
        print(f"[warn] {len(failures)}/{len(paths)} 张未处理成功，详见返回值的 failures")

    return {"total": len(paths), "done": done, "failures": failures}


def normalize_labels(src_path: str) -> dict:
    """把标签坐标夹紧到 (0, 1) 开区间内：>= 1.0 改写为 0.99999，<= 0 的负值改写为 0.00001。

    负坐标同样会让 ultralytics 加载标签时出错，旧版只夹上界、下界原样留着，这里补齐。
    逐列检查坐标（跳过类别列），未越界的值保持原文本；空行与非数值列原样保留；
    NaN 没法夹，只报出来。只有确实发生改动才就地覆盖写，且只重写发生改动的行。

    返回 {"files", "changed_files", "clipped", "nan_values"}。
    """

    files = glob.glob(os.path.join(src_path, "*.txt"))
    changed_files = 0
    clipped = 0
    nan_values: list[tuple[str, int]] = []

    for path in tqdm(files):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        changed = False
        for index, line in enumerate(lines):
            parts = line.split()
            if not parts:
                continue  # 空行原样保留

            new_parts = list(parts)
            touched = False
            for i in range(1, len(parts)):  # 第 0 列是类别 id，不参与坐标夹紧
                try:
                    value = float(parts[i])
                except ValueError:
                    continue  # 非数值列原样保留
                if math.isnan(value):
                    nan_values.append((path, index + 1))
                    continue
                if value >= 1.0:
                    new_parts[i] = "0.99999"
                    touched = True
                elif value < 0.0:
                    new_parts[i] = "0.00001"
                    touched = True

            if touched:
                # 只重写这一行；同文件里其它行保持原字节，不再被顺手整篇重排
                lines[index] = " ".join(new_parts) + "\n"
                clipped += sum(
                    1 for i in range(1, len(parts)) if new_parts[i] != parts[i]
                )
                changed = True

        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            changed_files += 1

    if nan_values:
        print(
            f"[warn] {len(nan_values)} 行坐标是 NaN，无法夹紧（文件, 行号）："
            f"{nan_values[:5]}{' …' if len(nan_values) > 5 else ''}"
        )

    return {
        "files": len(files),
        "changed_files": changed_files,
        "clipped": clipped,
        "nan_values": nan_values,
    }


# 第一列（类别 ID）之外的部分原样保留，所以只替换行首那一个非空白片段
_FIRST_COL = re.compile(r"(\s*)(\S+)(.*)", re.DOTALL)


def _id_key(value, label: str) -> str:
    """YOLO txt 的类别 ID 只能是非负整数：能转就转成规范字符串，转不动就把话说清楚。

    旧版直接 int(origin_id)，传个 "car" 进来抛的是 "invalid literal for int()"，
    看不出到底哪个参数、该传什么。
    """

    text = str(value).strip()
    try:
        num = int(float(text))
    except ValueError:
        raise ValueError(
            f"{label}只能是非负整数或 \"all\"，收到 {value!r}"
        ) from None

    if num < 0:
        raise ValueError(f"{label}不能为负数，收到 {value!r}")

    return str(num)


def _line_id_key(col: str) -> str:
    """把标签里已有的类别 ID 归一成比较用的键，让文件里写的 1.0 也能被 ID 1 命中。"""

    try:
        return str(int(float(col)))
    except ValueError:
        return col


def _replace_labels_core(txt_path: str, origin_key, target_key: str, replace_all: bool):
    """替换核心：读文件 -> 逐行改写 -> 有变化才就地覆盖写。

    返回 (changed, replaced_count, matched)。I/O 异常不在此捕获，向上抛给调用方——
    吞掉异常会让「没权限读」和「这个文件本来就没这个类别」都变成 (False, 0)，分不清。
    matched 是"命中 origin_key 的有效行数"，与"是否发生改动"分开统计——
    两者只在 origin_id == target_id 时才可能不一致。
    """

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    changed = False
    replaced_count = 0
    matched = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            # 空行或格式不合法（不足 5 列），整行原样保留
            new_lines.append(line)
            continue

        # 第一个元素是类别 ID；replace_all 时全部命中，否则只命中与 origin_id 相等的行
        if replace_all or _line_id_key(parts[0]) == origin_key:
            matched += 1
            if parts[0] != target_key:  # 已经是目标类别的不计入改动
                hit = _FIRST_COL.match(line)
                new_lines.append(hit.group(1) + target_key + hit.group(3))
                changed = True
                replaced_count += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if changed:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changed, replaced_count, matched


def _resolve_ids(origin_id, target_id):
    """校验并归一两个 ID，返回 (replace_all, origin_key, target_key)。"""

    replace_all = isinstance(origin_id, str) and origin_id.strip().lower() == "all"
    origin_key = None if replace_all else _id_key(origin_id, "原始类别 ID")
    target_key = _id_key(target_id, "目标类别 ID")

    return replace_all, origin_key, target_key


def replace_labels_with_target(
    txt_path: str, origin_id, target_id: int, warn_if_no_match: bool = False
):
    """把标签文件里「指定类别」每行的类别 ID 改为 target_id，保留坐标列。

    origin_id        : 要被替换的原始类别 ID；传字符串 "all"（不区分大小写）时替换所有类别。
                       传其它值时必须能当非负整数用，否则直接报错（旧版是 int() 抛一句
                       看不懂的 "invalid literal for int()"）。
    target_id        : 目标类别 ID。
    warn_if_no_match : True 时，若整个文件一行都没命中 origin_id（即根本不存在该类别）
                       就打印告警。origin_id="all" 时不会告警（全量替换无所谓命中）。

    例：把 1 改成 2   -> replace_labels_with_target(p, 1, 2)
        把所有改成 2 -> replace_labels_with_target(p, "all", 2)
        顺带查漏     -> replace_labels_with_target(p, 1, 2, warn_if_no_match=True)

    只有确实发生改动时才就地覆盖写，且只重写命中的行，未命中的行字节不变。
    返回 (changed, num_lines_replaced)；文件读不了会抛异常，不再静默返回 (False, 0)。
    """

    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"标签文件不存在或不是文件：{txt_path}")

    replace_all, origin_key, target_key = _resolve_ids(origin_id, target_id)

    changed, replaced_count, matched = _replace_labels_core(
        txt_path, origin_key, target_key, replace_all
    )

    if warn_if_no_match and not replace_all and matched == 0:
        tqdm.write(
            f"⚠️ 未命中类别 {origin_id}："
            f"{os.path.basename(txt_path)} 中没有任何该类别标签"
        )

    return changed, replaced_count


def batch_replace_labels(
    folder: str, origin_id, target_id: int, warn_if_no_match: bool = False
):
    """对文件夹下所有 .txt 标签批量执行类别 ID 替换（不递归子文件夹）。

    origin_id / target_id 语义与 replace_labels_with_target 一致，传 origin_id="all" 即替换全部类别。
    warn_if_no_match=True 时，整文件一行都没命中就逐条打印告警。

    返回 dict: {'total', 'changed_files', 'replaced_lines', 'no_match_files', 'failed'}
    no_match_files 统计"一行都没命中该类别"的文件数（origin_id="all" 时恒为 0）；
    failed 是读不了的 (路径, 错误) 清单——单个文件出错不影响其余文件。
    """

    if not os.path.isdir(folder):
        raise NotADirectoryError(f"文件夹不存在或不是一个目录：{folder}")

    replace_all, origin_key, target_key = _resolve_ids(origin_id, target_id)

    txt_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))
    if not txt_files:
        return {
            "total": 0,
            "changed_files": 0,
            "replaced_lines": 0,
            "no_match_files": 0,
            "failed": [],
        }

    changed_files = 0
    total_replaced = 0
    no_match_files = 0
    failed: list[tuple[str, str]] = []

    for filename in tqdm(txt_files, desc="处理进度", unit="个"):
        txt_path = os.path.join(folder, filename)

        try:
            changed, count, matched = _replace_labels_core(
                txt_path, origin_key, target_key, replace_all
            )
        except Exception as exc:
            tqdm.write(f"❌ 处理 {filename} 失败: {exc}")
            failed.append((txt_path, str(exc)))
            continue

        if changed:
            changed_files += 1
            total_replaced += count

        # 整文件没命中该类别，单独计数（origin_id="all" 时不可能发生）
        if not replace_all and matched == 0:
            no_match_files += 1
            if warn_if_no_match:
                tqdm.write(f"⚠️ 未命中类别 {origin_id}：{filename} 中没有任何该类别标签")

    return {
        "total": len(txt_files),
        "changed_files": changed_files,
        "replaced_lines": total_replaced,
        "no_match_files": no_match_files,
        "failed": failed,
    }


def collect_images(root, recursive: bool, image_exts: set | None = None) -> list[Path]:
    """收集 root 下所有图片文件，返回排序后的 Path 列表。

    root 收 str / os.PathLike / Path 都行（旧版只认 Path，按 README 的示例传字符串会
    直接 AttributeError），路径里的 ~ 也会展开。
    """

    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"图片文件夹不存在或不是目录：{root}")

    exts = IMAGE_EXTS if image_exts is None else image_exts
    it = root.rglob("*") if recursive else root.glob("*")

    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)


def fill_empty_labels(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    label_ext: str = ".txt",
    dry_run: bool = False,
    verbose: bool = False,
    image_exts: set | None = None,
) -> dict:
    """为缺失标签的图片生成空标签文件（YOLO 背景图常用）。

    参数
    ----
    images_dir : 目标图片文件夹
    labels_dir : 图片标签文件夹（用于判断标签是否已存在）
    output_dir : 补全标签目标文件夹，None 表示与 labels_dir 相同（就地补全）
    recursive  : 是否递归遍历子文件夹（保持目录结构）
    label_ext  : 标签文件扩展名
    dry_run    : True 时只预览不写文件
    verbose    : 是否打印每个新建标签文件的路径
    image_exts : 视为图片的扩展名集合，None 表示使用模块默认 IMAGE_EXTS

    返回
    ----
    dict: {'total', 'existed', 'created', 'failed', 'failures'}
    """

    img_root = Path(images_dir).expanduser().resolve()
    lbl_root = Path(labels_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve() if output_dir else lbl_root

    if not label_ext.startswith("."):
        label_ext = "." + label_ext

    # ---------- 输入校验 ----------
    if not img_root.is_dir():
        raise FileNotFoundError(f"图片文件夹不存在或不是文件夹：{img_root}")

    if not lbl_root.is_dir():
        print(f"[提示] 标签文件夹不存在，将视为所有图片均缺失标签：{lbl_root}")

    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    # ---------- 收集图片 ----------
    images = collect_images(img_root, recursive, image_exts)

    print(f"图片文件夹  : {img_root}")
    print(f"标签文件夹  : {lbl_root}")
    print(f"输出文件夹  : {out_root}" + ("（预览模式）" if dry_run else ""))
    print(f"标签扩展名  : {label_ext}")
    print(f"递归遍历    : {'是' if recursive else '否'}")
    print(f"共发现图片  : {len(images)} 张\n")

    if not images:
        print("[提示] 未找到任何图片文件，程序结束。")
        return {"total": 0, "existed": 0, "created": 0, "failed": 0, "failures": []}

    # ---------- 主循环 ----------
    created = 0
    existed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    created_list: list[Path] = []

    pbar = tqdm(
        images,
        desc="补全空标签",
        unit="img",
        ncols=100,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    for img_path in pbar:
        rel = img_path.relative_to(img_root)
        rel_label = rel.parent / (img_path.stem + label_ext)

        check_path = lbl_root / rel_label  # 判断标签是否已存在
        target_path = out_root / rel_label  # 实际写入位置

        if check_path.exists() or target_path.exists():
            existed += 1
        else:
            if dry_run:
                created += 1
                created_list.append(target_path)
            else:
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.touch(exist_ok=True)  # 生成 0 字节空文件
                except OSError as exc:
                    failed += 1
                    failures.append((str(target_path), str(exc)))
                else:
                    created += 1
                    created_list.append(target_path)

        pbar.set_postfix(新建=created, 已有=existed, 失败=failed, refresh=False)

    pbar.close()

    # ---------- 结果输出 ----------
    if verbose and created_list:
        print("\n新建的标签文件：")
        for p in created_list:
            print(f"  {p}")

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"图片总数        : {len(images)}")
    print(f"已有标签(跳过)  : {existed}")
    print(f"{'待新建' if dry_run else '已新建空标签'}: {created}")
    print(f"失败            : {failed}")
    if failures:
        print("\n失败明细（最多显示 10 条）：")
        for path, err in failures[:10]:
            print(f"  {path} -> {err}")

    if dry_run and created:
        print(
            "\n[提示] 当前为预览模式，未写入任何文件；把 dry_run 改为 False 即可实际执行。"
        )

    return {
        "total": len(images),
        "existed": existed,
        "created": created,
        "failed": failed,
        "failures": failures,
    }


def _count_txt(path: Path) -> Counter:
    """统计 YOLO txt 标签：每行首列即类别 ID。

    只跳过空行，不校验列数——计数场景下"这行属于哪个类别"和坐标是否合法无关。
    """

    counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                counter[parts[0]] += 1
    return counter


def _count_xml(path: Path) -> Counter:
    """统计 VOC xml 标签：每个 object 计一次数，类别取 name（缺失时退到 label）。

    标签名比较会忽略命名空间前缀；确实取不到类别名的 object 直接跳过，不计入任何类别。
    """

    counter = Counter()
    root = ET.parse(path).getroot()

    for elem in root.iter():
        if elem.tag.split("}")[-1] != XML_OBJECT_TAG:
            continue

        name = ""
        for child in elem:
            if child.tag.split("}")[-1] in XML_NAME_TAGS:
                name = (child.text or "").strip()
                if name:
                    break

        if name:
            counter[name] += 1

    return counter


def _display_width(text) -> int:
    """按终端显示宽度算长度：CJK / 全角字符占 2 列。

    类名是中文时，len() 会把"裂缝"算成 2 而终端占 4 列，
    直接用 len() 对齐会让表格错位。
    """

    return sum(
        2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(text)
    )


def _apply_names(key: str, names) -> str:
    """把类别 ID 映射成 "ID:名称" 展示；没给映射表或未收录时原样返回。

    映射表的键可能是 int（yaml 加载的 names）也可能是 str（手写的 dict），两种都认。
    非数字键（xml 的类别名）不做映射。保留 ID 前缀是为了回溯时能对上原始标签。
    """

    if not names:
        return key

    for probe in (key, int(key) if key.isdigit() else None):
        if probe is not None and probe in names:
            return f"{key}:{names[probe]}"

    return key


def count_labels(
    labels_dir: str | Path,
    recursive: bool = False,
    label_exts: tuple = LABEL_EXTS,
    names: dict | None = None,
    verbose: bool = False,
) -> dict:
    """统计标签文件夹里各类别的标注数量，支持 YOLO txt 与 VOC xml。

    注意：txt 统计出来的是类别 ID（"0"、"1"…），xml 统计出来的是类别名（"car"…）。
    同一目录混放两种格式时，counts 里会同时出现数字与名字——
    想按来源分开看，用返回值里的 by_format。

    参数
    ----
    labels_dir : 标签文件夹
    recursive  : 是否递归遍历子文件夹
    label_exts : 参与统计的扩展名，默认 (".txt", ".xml")
    names      : 类别 ID -> 名称的映射，如 {0: "裂缝", 1: "剥落"}，也可直接给一个列表
                 （按位置当 0、1、2… 的映射，ultralytics 的 yaml names 就有这种）。
                 给了之后数字类别的键显示成 "0:裂缝"（保留 ID 便于回溯原始标签）；
                 未收录的 ID 原样显示，xml 的类别名不受影响。
    verbose    : True 时逐个打印每个文件的类别计数

    返回
    ----
    dict: {
        'files'        扫描到的标签文件数,
        'parsed_files' 成功解析的文件数,
        'labels'       标注总数,
        'counts'       {类别: 数量}，按数量降序,
        'by_format'    {扩展名: {类别: 数量}}，只含出现过的格式,
        'failed'       [(路径, 错误信息), ...],
    }
    """

    root = Path(labels_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"标签文件夹不存在或不是文件夹：{root}")

    # 列表形式的 names 按位置补成 {0: ..., 1: ...}，否则 _apply_names 会把值当键去查
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))

    exts = tuple(e if e.startswith(".") else "." + e for e in label_exts)
    it = root.rglob("*") if recursive else root.glob("*")
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)

    total: Counter = Counter()
    by_format: dict[str, Counter] = {e: Counter() for e in exts}
    failed: list[tuple[str, str]] = []

    pbar = tqdm(
        files,
        desc="统计标签",
        unit="个",
        ncols=100,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    for path in pbar:
        ext = path.suffix.lower()
        try:
            counter = _count_xml(path) if ext == ".xml" else _count_txt(path)
        except Exception as exc:
            # 单个文件解析失败不影响整体统计，记下来最后统一展示
            failed.append((str(path), str(exc)))
            continue

        total.update(counter)
        by_format[ext].update(counter)

        if verbose:
            shown_file = {_apply_names(k, names): v for k, v in counter.items()}
            print(f"  {path.name}: {shown_file}")

        pbar.set_postfix(类别=len(total), 标注=sum(total.values()), refresh=False)

    pbar.close()

    # ---------- 汇总输出 ----------
    # Counter 内部保留原始键，只在展示与返回值里套上 names 映射
    shown: Counter = Counter()
    for cls, n in total.items():
        shown[_apply_names(cls, names)] += n

    print("\n" + "=" * 60)
    print("类别计数结果")
    print("=" * 60)
    print(f"标签文件夹      : {root}")
    print(f"扫描文件数      : {len(files)}")
    print(f"成功解析        : {len(files) - len(failed)}")
    print(f"标注总数        : {total.total()}")
    print(f"类别总数        : {len(shown)}")
    if names:
        print(f"类别命名        : 已应用 {len(names)} 条 ID -> 名称映射")

    if shown:
        print("-" * 60)
        width = max(_display_width(c) for c in shown)
        for cls, n in shown.most_common():
            label = str(cls)
            print(f"  {label}{' ' * (width - _display_width(label))}  {n}")

    if failed:
        print("-" * 60)
        print(f"解析失败 {len(failed)} 个（最多显示 10 条）：")
        for path, err in failed[:10]:
            print(f"  {path} -> {err}")

    print("=" * 60)

    by_format_shown: dict[str, dict] = {}
    for ext, counter in by_format.items():
        if not counter:
            continue
        bucket: Counter = Counter()
        for cls, n in counter.items():
            bucket[_apply_names(cls, names)] += n
        by_format_shown[ext] = dict(bucket.most_common())

    return {
        "files": len(files),
        "parsed_files": len(files) - len(failed),
        "labels": total.total(),
        "counts": dict(shown.most_common()),
        "by_format": by_format_shown,
        "failed": failed,
    }


# 用法示例：
#     from ObjectDetection import count_labels, fill_empty_labels, replace_labels_with_target
#     count_labels("./datasets/xxx/labels")                      # 统计 txt + xml
#     count_labels("./datasets/xxx/labels", recursive=True)      # 含子文件夹
#     count_labels("./datasets/xxx/labels", label_exts=(".xml",))  # 只统计 xml
#     count_labels("./datasets/xxx/labels", names={0: "裂缝", 1: "剥落"})  # 给 ID 配上名字
#     fill_empty_labels("./datasets/xxx/images", "./datasets/xxx/labels", dry_run=True)
#     replace_labels_with_target("./meta/a.txt", 1, 2)       # 把类别 1 改成 2
#     replace_labels_with_target("./meta/a.txt", "all", 2)   # 所有类别都改成 2
#     batch_replace_labels("./datasets/交通标志&锥桶/meta/", 1, 2, warn_if_no_match=True)
