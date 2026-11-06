"""视频抽帧：按固定间隔从视频中提取图片帧，支持单文件与递归批量处理。

函数清单：
    extract_frames    主入口：批量抽帧 + 配置打印 + 结果报告，返回统计 dict
    process_video     抽取单个视频的帧，返回该视频的统计信息
    collect_videos    从文件或文件夹中收集视频文件
    print_config / print_report / print_summary   输出配置与统计报告
    build_encode_params / write_image / fmt_duration / log   内部工具

依赖：opencv-python、tqdm。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

try:
    import cv2
except ImportError:
    raise ImportError("缺少依赖 opencv-python，请执行: pip install opencv-python")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("缺少依赖 tqdm，请执行: pip install tqdm")


VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".flv", ".webm", ".wmv"}
SEP_THICK = "=" * 68
SEP_THIN = "-" * 66


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    """与 tqdm 进度条安全共存的打印。"""

    tqdm.write(msg)


def fmt_duration(seconds: float) -> str:
    """把秒格式化成 HH:MM:SS。"""

    if seconds <= 0:
        return "未知"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_encode_params(ext: str, quality: int) -> List[int]:
    """根据图片格式构造 OpenCV 编码参数。"""

    q = max(1, min(100, int(quality)))
    if ext in ("jpg", "jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, q]
    if ext == "webp":
        return [cv2.IMWRITE_WEBP_QUALITY, q]
    if ext == "png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    return []


def write_image(path: Path, frame, ext: str, params: List[int]) -> bool:
    """保存图片，兼容中文路径。"""

    try:
        ok, buf = cv2.imencode(f".{ext}", frame, params)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception as exc:
        log(f"  [警告] 保存失败 {path}: {exc}")
        return False


def collect_videos(input_path: Path, recursive: bool) -> List[Path]:
    """从文件或文件夹中收集视频文件。"""

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        it = input_path.rglob("*") if recursive else input_path.glob("*")
        return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    return []


# --------------------------------------------------------------------------- #
# 配置打印
# --------------------------------------------------------------------------- #
def print_config(
    input_path: Path,
    out_root: Path,
    videos: List[Path],
    every_n_frames: int,
    image_ext: str,
    image_quality: int,
    flat_output: bool,
    filename_prefix: bool,
    recursive: bool,
) -> None:
    """打印本次抽帧的输入配置。"""

    print(SEP_THICK)
    print("  视频抽帧 —— 输入配置")
    print(SEP_THICK)
    print(f"  输入路径   : {input_path.resolve()}")
    print(f"  输出文件夹 : {out_root.resolve()}")
    print(f"  截帧间隔   : 每 {every_n_frames} 帧取 1 张")
    quality_info = (
        f"（质量 {image_quality}）"
        if image_ext.lower() in ("jpg", "jpeg", "webp")
        else ""
    )
    print(f"  图片格式   : {image_ext.lower()}{quality_info}")
    print(
        f"  目录结构   : {'所有图片同一文件夹' if flat_output else '每个视频一个子文件夹'}"
    )
    print(f"  文件前缀   : {'视频名_' if filename_prefix else '无'}")
    print(f"  递归搜索   : {'是' if recursive else '否'}")
    print(f"  视频数量   : {len(videos)}")
    print(SEP_THICK)
    print("  待处理视频：")
    for v in videos:
        print(f"    - {v}")
    print(SEP_THICK)


# --------------------------------------------------------------------------- #
# 核心：单个视频抽帧
# --------------------------------------------------------------------------- #
def process_video(
    video_path: Path,
    out_dir: Path,
    every_n_frames: int,
    image_ext: str,
    image_quality: int,
    filename_prefix: bool,
) -> Optional[dict]:
    """抽取单个视频的帧，返回统计信息；失败返回 None。"""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log(f"  [错误] 无法打开视频: {video_path}")
        return None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:
        fps, total, width, height = 0.0, 0, 0, 0

    duration = total / fps if fps > 0 else 0.0
    every = max(1, int(every_n_frames))
    ext = image_ext.lower()
    params = build_encode_params(ext, image_quality)

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    desc = stem if len(stem) <= 28 else stem[:25] + "..."

    pbar = tqdm(
        total=total if total > 0 else None,
        desc=desc,
        unit="帧",
        ncols=100,
        dynamic_ncols=True,
        leave=True,
    )

    idx = 0
    saved = 0
    t0 = time.time()

    try:
        while True:
            if not cap.grab():
                break

            if idx % every == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    name = (
                        f"{stem}_{idx:06d}.{ext}"
                        if filename_prefix
                        else f"{idx:06d}.{ext}"
                    )
                    if write_image(out_dir / name, frame, ext, params):
                        saved += 1

            idx += 1
            pbar.update(1)
    except KeyboardInterrupt:
        log("\n  [中断] 用户手动终止当前视频。")
    finally:
        if pbar.total is not None and pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()
        cap.release()

    return {
        "name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total": total if total > 0 else idx,
        "read": idx,
        "saved": saved,
        "out_dir": out_dir,
        "elapsed": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def print_report(r: dict) -> None:
    """打印单个视频的处理结果。"""

    log("  " + SEP_THIN)
    log(f"  ✔ 视频       : {r['name']}")
    log(f"    分辨率     : {r['width']} x {r['height']}")
    log(f"    帧率 / 时长: {r['fps']:.2f} fps  /  {fmt_duration(r['duration'])}")
    log(f"    总帧数     : {r['total']}")
    log(f"    提取图片   : {r['saved']} 张")
    log(f"    保存位置   : {r['out_dir'].resolve()}")
    log(f"    耗时       : {r['elapsed']:.2f} s")
    log("  " + SEP_THIN)


def print_summary(results: List[dict], failures: List[str], out_root: Path) -> None:
    """打印全部视频的汇总统计。"""

    total_saved = sum(r["saved"] for r in results)
    total_read = sum(r["read"] for r in results)
    total_time = sum(r["elapsed"] for r in results)

    print()
    print(SEP_THICK)
    print("  全部完成")
    print(SEP_THICK)
    print(f"  成功视频数   : {len(results)}")
    if failures:
        print(f"  失败视频数   : {len(failures)}")
        for f in failures:
            print(f"      × {f}")
    print(f"  扫描总帧数   : {total_read}")
    print(f"  提取图片总数 : {total_saved} 张")
    print(f"  图片保存位置 : {out_root.resolve()}")
    print(f"  总耗时       : {total_time:.2f} s")
    print(SEP_THICK)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def extract_frames(
    input_path: str | Path,
    output_dir: str | Path,
    every_n_frames: int = 10,
    image_ext: str = "jpg",
    image_quality: int = 100,
    recursive: bool = True,
    flat_output: bool = True,
    filename_prefix: bool = True,
) -> dict:
    """按固定间隔从视频中抽帧并保存为图片。

    参数
    ----
    input_path      : 输入视频文件路径，或包含视频的文件夹
    output_dir      : 图片保存文件夹
    every_n_frames  : 每多少帧截取一张（1 = 每一帧都保存）
    image_ext       : 图片格式：jpg / jpeg / png / bmp / webp
    image_quality   : jpg / webp 的质量（1-100），png 忽略此项
    recursive       : 输入是文件夹时，是否递归搜索子文件夹
    flat_output     : True = 所有图片放同一文件夹；False = 每个视频一个子文件夹
    filename_prefix : 图片名是否带视频名前缀（防止重名覆盖）

    返回
    ----
    dict: {'results', 'failures', 'out_root', 'total_saved'}

    异常
    ----
    ValueError        : every_n_frames < 1
    FileNotFoundError : 输入路径不存在，或未找到任何视频文件
    """

    if every_n_frames < 1:
        raise ValueError("every_n_frames 必须 >= 1")

    in_path = Path(input_path).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {in_path.resolve()}")

    videos = collect_videos(in_path, recursive)
    if not videos:
        raise FileNotFoundError(
            f"未找到视频文件（支持: {', '.join(sorted(VIDEO_EXTS))}）: {in_path.resolve()}"
        )

    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 打印输入配置 ----
    print_config(
        in_path,
        out_root,
        videos,
        every_n_frames,
        image_ext,
        image_quality,
        flat_output,
        filename_prefix,
        recursive,
    )

    results: List[dict] = []
    failures: List[str] = []

    for i, video in enumerate(videos, 1):
        out_dir = out_root if flat_output else out_root / video.stem
        log(f"\n[{i}/{len(videos)}] 开始处理: {video.name}")

        result = process_video(
            video, out_dir, every_n_frames, image_ext, image_quality, filename_prefix
        )
        if result is None:
            failures.append(str(video))
            continue

        results.append(result)
        print_report(result)

    print_summary(results, failures, out_root)

    return {
        "results": results,
        "failures": failures,
        "out_root": out_root,
        "total_saved": sum(r["saved"] for r in results),
    }


# 用法示例：
#     from ObjectDetection import extract_frames
#     extract_frames("./datasets/样例视频", "./datasets/样例视频/frames", every_n_frames=10)
