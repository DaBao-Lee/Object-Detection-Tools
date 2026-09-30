"""模型权重定位：取训练产物中的 best.pt / last.pt。"""

import glob
import os


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
