import numpy as np
from lxml import etree
from enum import Enum
from glob import glob
from tqdm import tqdm
from typing import Union
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os, shutil, random, re, yaml, cv2, json, colorama

def create_dirs(path: str):
    """
    创建指定路径的目录。如果目录已存在，提示用户是否覆盖。
    :param path: 要创建的目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        res = input(f"{path} already exists, wheater to overwrite? [y/n]")
        if res == "y":
            shutil.rmtree(path)
            os.makedirs(path)
            print("Overwrite Success.")
        else: print("Overwrite Insuccess.")

def rm_dirs(path: str):
    """
    删除指定路径的目录及其所有内容。
    :param path: 要删除的目录路径
    """
    if os.path.exists(path): shutil.rmtree(path)

def fetch_specific_files(path: Union[str, list], file_type: Union[str, tuple]="*"):
    """
    获取指定路径下所有特定类型的文件。
    :param path: 要搜索的目录路径，必须以 / 结尾
    :param file_type: 文件类型，默认为所有文件
    :return: 符合条件的文件列表
    """
    
    files = []
    if isinstance(path, str):
        assert os.path.exists(path), "Path does not exist."
        assert path.endswith("/"), "Path must end with /"

        if isinstance(file_type, tuple):
            for typing in file_type:
                files.extend(glob(f"{path}/*.{typing}", recursive=True))
        else: files = glob(f"{path}/*.{file_type}", recursive=True)
        
    else: files.extend([x for x in path if x.endswith(f".{file_type}")])
    
    return files
def process_files(src: Union[str, list], dst: str, file_type: str="*", mode: int=0):
    """
    根据模式处理文件，可以复制或删除文件。
    :param src: 源文件路径或文件列表
    :param dst: 目标目录路径
    :param file_type: 文件类型，默认为所有文件
    :param mode: 处理模式，0 表示复制，1 表示删除
    """
    assert os.path.exists(dst), "Destination path does not exist."
    files = src
    if type(src) == str:
        assert os.path.exists(src), "Source path does not exist."
        files = fetch_specific_files(src, file_type=file_type)

    if mode == 0:
        for file in tqdm(files): shutil.copy(file, dst)
    elif mode == 1:
        for file in tqdm(files): os.unlink(file)

def ignore_something(mode: int=0):
    """
    忽略所有警告信息。
    param mode: 模式选择
    - 0: 表示忽略所有警告 
    - 1: 表示禁止打印DEBUG信息 适用于PaddleOCR
    - other: 表示忽略所有警告和禁止打印DEBUG信息
    """
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
