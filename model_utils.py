"""模型权重定位：取训练产物中的 best.pt / last.pt，按文件夹列出权重文件。"""

import glob
import os
from typing import List, Sequence, Union


def get_best_last_model(
    model_path: str = "./runs/detect/train/weights", index: int = None, mode: int = 0
) -> str:
    """返回权重路径。mode=0 取 best.pt，mode=1 取 last.pt（只实现了这两个分支）。

    传 index 则忽略 model_path，直接找 ./runs/detect/train{index}/weights。
    """

    if index is not None:
        model_path = f"./runs/detect/train{index}/weights"

    models = glob.glob(os.path.join(model_path, "*.pt"))
    assert models, f"{model_path} 未找到模型，请检查路径是否正确。"

    if mode == 0:
        path = next(x for x in models if "best" in os.path.basename(x))

    elif mode == 1:
        path = next(x for x in models if "last" in os.path.basename(x))

    return str(path)


WEIGHT_EXTS = (".pt", ".onnx")


# 用途：列出文件夹里的权重文件（默认 .pt / .onnx），按文件名排序，可选递归
def list_weights(
    folder: Union[str, os.PathLike],
    exts: Sequence[str] = WEIGHT_EXTS,
    recursive: bool = False,
) -> List[str]:

    folder = os.fspath(folder)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"权重文件夹不存在或不是目录：{folder}")

    # 调用方可能写 "pt" 也可能写 ".pt"，统一成带点的小写再比
    suffixes = tuple((e if e.startswith(".") else "." + e).lower() for e in exts)

    paths: List[str] = []
    for root, dirs, files in os.walk(folder):
        # os.walk 靠 dirs 决定是否下钻，就地清空 = 只扫当前这一层
        if not recursive:
            dirs.clear()
        paths.extend(os.path.join(root, name) for name in files)

    weights = [p for p in paths if os.path.splitext(p)[1].lower() in suffixes]
    weights.sort(key=lambda p: os.path.basename(p).lower())

    return weights
