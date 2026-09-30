import os
import shutil
from glob import glob
from typing import Union

from tqdm import tqdm


def create_dirs(path: str, overwrite: Union[bool, None] = None):

    # overwrite=None 保持原有交互确认；True / False 可跳过交互，供无头或服务端调用
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite is None:
        res = input(f"{path} already exists, wheater to overwrite? [y/n]")
        if res == "y":
            shutil.rmtree(path)
            os.makedirs(path)
            print("Overwrite Success.")
        else:
            print("Overwrite Insuccess.")
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)


def rm_dirs(path: str):

    if os.path.exists(path):
        shutil.rmtree(path)


def fetch_specific_files(path: Union[str, list], file_type: Union[str, tuple] = "*"):

    files = []
    if isinstance(path, str):
        assert os.path.exists(path), "Path does not exist."

        # 兼容正 / 反斜杠结尾，Windows 路径不再被断言拦下
        path = path.rstrip("/\\")

        if isinstance(file_type, tuple):
            for typing in file_type:
                files.extend(glob(os.path.join(path, f"*.{typing}"), recursive=True))
        else:
            files = glob(os.path.join(path, f"*.{file_type}"), recursive=True)

    elif file_type == "*":
        # 传列表且 file_type 为 "*" 时不过滤；原写法 endswith(".*") 必然返回空列表
        files.extend(path)
    else:
        exts = (file_type,) if isinstance(file_type, str) else tuple(file_type)
        files.extend([x for x in path if x.endswith(tuple(f".{e}" for e in exts))])

    return files


def process_files(src: Union[str, list], dst: str, file_type: str = "*", mode: int = 0):

    assert os.path.exists(dst), "Destination path does not exist."
    files = src
    if type(src) == str:
        assert os.path.exists(src), "Source path does not exist."
        files = fetch_specific_files(src, file_type=file_type)

    if mode == 0:
        for file in tqdm(files):
            shutil.copy(file, dst)
    elif mode == 1:
        for file in tqdm(files):
            os.unlink(file)


def ignore_something(mode: int = 0):

    if mode == 0:
        import warnings

        warnings.filterwarnings("ignore")
    elif mode == 1:
        import logging

        logging.disable(logging.DEBUG)
    else:
        import warnings, logging

        warnings.filterwarnings("ignore")
        logging.disable(logging.DEBUG)


def add_suffix_to_files(folder_path: str, suffix: str):

    """给文件夹内每个文件追加后缀（小尾巴），不改扩展名。

    只处理 folder_path 顶层文件，不递归子文件夹。注意：就地重命名，不可逆；
    目标名已存在时跳过该文件，单文件失败只告警、不中断整体流程。
    """

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    for filename in tqdm(files, desc="重命名进度", unit="个"):
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{ext}"

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # 新名与旧名相同则跳过
        if old_path == new_path:
            continue

        # 目标文件已存在时跳过并告警，避免覆盖
        if os.path.exists(new_path):
            tqdm.write(f"⚠️ 警告: {new_filename} 已存在，跳过 {filename}")
            continue

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            tqdm.write(f"❌ 重命名 {filename} 失败: {e}")
