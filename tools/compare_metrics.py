from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "Metric",
    "METRICS",
    "list_metrics",
    "resolve_metrics",
    "expand_csv_inputs",
    "load_results_csv",
    "read_run_label",
    "smooth_series",
    "last_n_stats",
    "plot_curves",
    "compare_metrics",
]


# --------------------------------------------------------------------------- #
# 1. 指标注册表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metric:
    """一个可对比的指标。

    key     简写名，供 resolve_metrics 选择
    columns 参与求平均的列名（元组长度 > 1 时取均值，用于三个 pg 学习率合并成一个 lr）
    label   图例 / 标题显示名
    group   分组，仅用于 list_metrics 分类展示
    higher_is_better  True 越高越好 / False 越低越好 / None 中性
    """

    key: str
    columns: tuple[str, ...]
    label: str
    group: str
    higher_is_better: bool | None = None

    @property
    def primary_column(self) -> str:
        return self.columns[0]


METRICS: tuple[Metric, ...] = (
    # 训练损失
    Metric("box_loss", ("train/box_loss",), "Box Loss (train)", "训练损失", False),
    Metric("cls_loss", ("train/cls_loss",), "Cls Loss (train)", "训练损失", False),
    Metric("dfl_loss", ("train/dfl_loss",), "DFL Loss (train)", "训练损失", False),
    # 验证指标
    Metric("precision", ("metrics/precision(B)",), "Precision", "验证指标", True),
    Metric("recall", ("metrics/recall(B)",), "Recall", "验证指标", True),
    Metric("mAP50", ("metrics/mAP50(B)",), "mAP@0.5", "验证指标", True),
    Metric("mAP50-95", ("metrics/mAP50-95(B)",), "mAP@0.5:0.95", "验证指标", True),
    # 验证损失
    Metric("val/box_loss", ("val/box_loss",), "Box Loss (val)", "验证损失", False),
    Metric("val/cls_loss", ("val/cls_loss",), "Cls Loss (val)", "验证损失", False),
    Metric("val/dfl_loss", ("val/dfl_loss",), "DFL Loss (val)", "验证损失", False),
    # 训练过程
    Metric("time", ("time",), "Time (h)", "训练过程", None),
    Metric("lr/pg0", ("lr/pg0",), "LR (pg0)", "训练过程", None),
    Metric("lr/pg1", ("lr/pg1",), "LR (pg1)", "训练过程", None),
    Metric("lr/pg2", ("lr/pg2",), "LR (pg2)", "训练过程", None),
    Metric(
        "lr",
        ("lr/pg0", "lr/pg1", "lr/pg2"),
        "LR (avg)",
        "训练过程",
        None,
    ),
)

_METRIC_BY_KEY = {m.key: m for m in METRICS}

# 无障碍色板（Okabe-Ito），多序列叠加时区分度最高
_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
_CHINESE_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "WenQuanYi Micro Hei",
)


# --------------------------------------------------------------------------- #
# 2. 读取与解析
# --------------------------------------------------------------------------- #
def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """把列名两侧空白和残留的(B) 空格清掉，ultralytics 不同版本写法不一致。"""
    rename = {}
    for col in frame.columns:
        name = re.sub(r"\s+", "", str(col).strip())
        if name != col:
            rename[col] = name
    return frame.rename(columns=rename) if rename else frame


def load_results_csv(csv_path: str | Path) -> pd.DataFrame:
    """读入单个 results.csv，做列名清洗、数值化、epoch 排序。"""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"results.csv 不存在：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = _clean_columns(frame)
    if "epoch" not in frame.columns:
        frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    for col in frame.columns:
        if col != "epoch":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values("epoch").reset_index(drop=True)


def expand_csv_inputs(
    paths: str | Path | Iterable[str | Path], recursive: bool = False
) -> list[Path]:
    """把 str / Path / 目录 / 列表 统一展开成 results.csv 路径列表（去重、保持顺序）。"""
    if isinstance(paths, (str, Path)):
        raw: list[str | Path] = [paths]
    else:
        raw = list(paths)

    found: list[Path] = []
    for item in raw:
        target = Path(item).expanduser()
        if target.is_dir():
            pattern = "**/results.csv" if recursive else "results.csv"
            found.extend(sorted(p for p in target.glob(pattern) if p.is_file()))
        elif target.is_file():
            found.append(target)
        else:
            raise FileNotFoundError(f"路径不存在：{target}")

    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in [p.resolve() for p in unique]:
            unique.append(path)
    if not unique:
        raise FileNotFoundError("没有找到任何 results.csv")
    return unique


def read_run_label(csv_path: str | Path, root: str | Path | None = None) -> str:
    """从路径里提取一个短标签作为图例名。

    root 给出时用「相对 root 的父目录名」（如 .../mosaic 10/improvement/ghost/results.csv
    -> ghost），否则退回 csv 所在目录名。
    """
    path = Path(csv_path).expanduser()
    if root is not None:
        try:
            rel_parent = path.resolve().parent.relative_to(
                Path(root).expanduser().resolve()
            )
        except ValueError:
            rel_parent = path.resolve().parent
        parts = rel_parent.parts
        if parts:
            return parts[-1]
    return path.parent.name or path.stem


def _metric_series(frame: pd.DataFrame, metric: Metric) -> pd.Series:
    """取出某指标对应的数值列（多列取均值），缺失列抛错。"""
    missing = [c for c in metric.columns if c not in frame.columns]
    if missing:
        raise KeyError(f"指标 {metric.key} 需要的列不存在：{missing}")
    series = frame[list(metric.columns)].mean(axis=1)
    return series.astype(float)


def smooth_series(values: Sequence[float] | np.ndarray, window: int = 0) -> np.ndarray:
    """滑动平均平滑；window<=1 时原样返回。"""
    array = np.asarray(values, dtype=float)
    if window is None or window <= 1:
        return array
    window = int(window)
    window = min(window, len(array))
    kernel = np.ones(window) / window
    padded = np.pad(array, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def last_n_stats(
    frame: pd.DataFrame, metric: Metric | str, last_n: int | None = 10
) -> dict[str, float | int | None]:
    """统计末 N 个epoch 的均值 / 标准差 / 最佳值。

    last_n=None 时统计全程（不推荐：close_mosaic 之前的指标受 mosaic 扰动）。
    """
    metric = _metric_by_key(metric)
    series = _metric_series(frame, metric).dropna()
    if series.empty:
        return {"mean": None, "std": None, "best": None, "n": 0, "start_epoch": None}
    tail = series if last_n is None else series.tail(int(last_n))
    values = tail.to_numpy(dtype=float)
    best = values.max() if metric.higher_is_better else values.min()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "best": float(best),
        "n": int(len(values)),
        "start_epoch": int(tail.index[0]) + 1,
    }


# --------------------------------------------------------------------------- #
# 3. 指标选择
# --------------------------------------------------------------------------- #
def available_metrics(frame: pd.DataFrame) -> list[Metric]:
    """当前 csv 实际拥有的指标（按注册表顺序）。"""
    return [m for m in METRICS if all(c in frame.columns for c in m.columns)]


def _metric_by_key(metric: Metric | str) -> Metric:
    if isinstance(metric, Metric):
        return metric
    if metric in _METRIC_BY_KEY:
        return _METRIC_BY_KEY[metric]
    raise KeyError(f"未知指标 key：{metric!r}，可选：{list(_METRIC_BY_KEY)}")


def list_metrics(
    csv_paths: str | Path | Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """罗列可选指标供用户选择。给 csv 时只显示该文件真实存在的列。

    返回一个 DataFrame，同时直接打印编号表；运行时凭编号调用 resolve_metrics。
    """
    rows: list[dict] = []
    for metric in METRICS:
        available = True
        if csv_paths is not None:
            available = any(
                all(c in load_results_csv(p).columns for c in metric.columns)
                for p in expand_csv_inputs(csv_paths)
            )
        arrow = {True: "↑ 越高越好", False: "↓ 越低越好", None: "— 参考"}[
            metric.higher_is_better
        ]
        rows.append(
            {
                "编号": len(rows) + 1,
                "key": metric.key,
                "列名": " / ".join(metric.columns),
                "显示名": metric.label,
                "分组": metric.group,
                "趋势": arrow,
                "当前csv可用": "是" if available else "否",
            }
        )

    table = pd.DataFrame(rows)
    print("=" * 78)
    print("可选指标清单（调用 resolve_metrics 时可用「编号」或 「key」）")
    print("=" * 78)
    print(table.to_string(index=False))
    print("-" * 78)
    print('示例：resolve_metrics("7")-> mAP@0.5:0.95；resolve_metrics(["3", "7"])')
    return table


def resolve_metrics(
    selection: str | int | Metric | Iterable[str | int | Metric] | None,
    csv_paths: str | Path | Iterable[str | Path] | None = None,
) -> list[Metric]:
    """把用户选择（编号 / key / 完整列名 / 模糊子串）解析成 Metric 列表。

    selection=None 时返回当前 csv 全部可用指标。
    """
    if selection is None:
        if csv_paths is None:
            return list(METRICS)
        frame = load_results_csv(expand_csv_inputs(csv_paths)[0])
        return available_metrics(frame)

    items: list[str | int | Metric] = (
        [selection] if isinstance(selection, (str, int, Metric)) else list(selection)
    )

    resolved: list[Metric] = []
    for item in items:
        metric = _match_one(item, csv_paths)
        if metric not in resolved:
            resolved.append(metric)
    return resolved


def _match_one(
    item: str | int | Metric, csv_paths: str | Path | Iterable[str | Path] | None
) -> Metric:
    if isinstance(item, Metric):
        return item
    if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit()):
        index = int(item) - 1
        if not 0 <= index < len(METRICS):
            raise IndexError(f"指标编号 {item} 越界，有效范围 1~{len(METRICS)}")
        return METRICS[index]

    text = str(item).strip()
    if text in _METRIC_BY_KEY:
        return _METRIC_BY_KEY[text]

    pool = (
        available_metrics(load_results_csv(expand_csv_inputs(csv_paths)[0]))
        if csv_paths
        else list(METRICS)
    )
    for metric in pool:  # 完整列名精确匹配
        if text in metric.columns:
            return metric
    for metric in pool:  # 模糊子串，忽略大小写与分隔符
        key = metric.key.replace("/", "").replace("-", "").lower()
        probe = text.replace("/", "").replace("-", "").lower()
        if probe and (probe == key or probe in key):
            return metric
    raise KeyError(f"无法识别指标 {item!r}，请先调用 list_metrics() 查看编号表")


# --------------------------------------------------------------------------- #
# 4. 绘图
# --------------------------------------------------------------------------- #
def _setup_style() -> None:
    """中文可用字体 + 负号正常显示 + 论文友好的白底样式。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CHINESE_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["font.size"] = 10


def _grid_shape(count: int) -> tuple[int, int]:
    ncols = 1 if count == 1 else 2 if count <= 4 else 3
    nrows = int(np.ceil(count / ncols))
    return nrows, ncols


def _finalize(fig: Figure, out_path: str | Path | None, dpi: int, show: bool) -> Figure:
    if out_path is not None:
        path = Path(out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[compare_metrics] 图已保存：{path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_curves(
    csv_paths: str | Path | Iterable[str | Path],
    metrics: str | int | Metric | Iterable[str | int | Metric] | None = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    last_n: int | None = 10,
    smooth_window: int = 0,
    std_band: bool = False,
    figsize: tuple[float, float] | None = None,
    out_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> Figure:
    """画指标曲线（唯一出图函数）。

    单个 csv -> 单次训练曲线；多个 csv -> 多次训练同指标叠加曲线（不画箱型/柱状图）。
    std_band=True 且 last_n 有效时，额外叠加末 N epoch 均值参考线。
    """
    _setup_style()
    paths = expand_csv_inputs(csv_paths)
    frames = {p: load_results_csv(p) for p in paths}
    targets = resolve_metrics(metrics, csv_paths=paths)
    if not targets:
        raise ValueError("没有可绘制的指标")

    if labels is None:
        labels = [read_run_label(p) for p in paths]
    names = _dedupe([str(x) for x in labels])

    nrows, ncols = _grid_shape(len(targets))
    base = figsize or (6.2 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=base, squeeze=False)
    flat_axes: list[Axes] = [ax for row in axes for ax in row]

    mode = "single run" if len(paths) == 1 else f"{len(paths)} runs"
    for ax, metric in zip(flat_axes, targets):
        for (path, frame), name, color in zip(frames.items(), names, _PALETTE):
            series = _metric_series(frame, metric)
            smoothed = smooth_series(series.to_numpy(), smooth_window)
            ax.plot(
                frame["epoch"],
                smoothed,
                color=color,
                linewidth=1.8,
                label=name,
            )
            if std_band and last_n:
                stats = last_n_stats(frame, metric, last_n)
                if stats["mean"] is not None:
                    ax.axhline(
                        stats["mean"],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                        label=(
                            f"{name} last{last_n}ep mean" if len(targets) == 1 else None
                        ),
                    )

        arrow = {True: " (↑)", False: " (↓)", None: ""}[metric.higher_is_better]
        ax.set_title(f"{metric.label}{arrow}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.primary_column)
        # 单文件只有一条线，不需要图例；多指标统一交给 figure 级图例
        if len(targets) == 1 and (len(paths) > 1 or (std_band and last_n)):
            ax.legend(fontsize=8, framealpha=0.9)

    for ax in flat_axes[len(targets) :]:
        ax.axis("off")

    # 多指标 + 多文件时用figure 级统一图例，避免每张子图重复
    if len(targets) > 1 and len(paths) > 1:
        handles, handle_labels = flat_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            handle_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncols=min(len(names), 5),
            fontsize=8.5,
            frameon=False,
        )

    subtitle = f"last {last_n} epochs" if last_n else "all epochs"
    fig.suptitle(
        title or f"Metrics Comparison  |  {mode}  |  {subtitle}",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(
        rect=(0, 0, 1, 0.90 if (len(targets) > 1 and len(paths) > 1) else 0.97)
    )
    return _finalize(fig, out_path, dpi, show)


def _dedupe(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}#{seen[name] + 1}")
    return out


# --------------------------------------------------------------------------- #
# 5. 总入口
# --------------------------------------------------------------------------- #
def compare_metrics(
    csv_paths: str | Path | Iterable[str | Path],
    metrics: str | int | Metric | Iterable[str | int | Metric] | None = None,
    labels: Sequence[str] | None = None,
    last_n: int | None = 10,
    smooth_window: int = 0,
    std_band: bool = False,
    title: str | None = None,
    root: str | Path | None = None,
    recursive: bool = True,
    out_dir: str | Path | None = None,
    out_name: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> dict[str, object]:
    """一站式指标对比

    Parameters
    ----------
    csv_paths    单个 / 多个 results.csv，也可以传目录（自动找 results.csv）
    metrics      指标选择：编号、key、完整列名，或它们的列表；None = csv 里全部可用指标
    labels       图例名；不给就用路径里的目录名
    last_n       统计口径的末 N 个 epoch（None = 全程，不推荐）
    smooth_window曲线滑动平均窗口，0 = 不平滑
    std_band     是否在曲线上叠加末 N epoch 的均值参考线
    root         提取图例名时的相对根目录
    recursive    传目录时是否递归查找子目录里的 results.csv
    out_dir/out_name  保存路径；给了 out_dir 就落盘为 {out_name}.png
    show         是否 plt.show()

    Returns
    -------
    dict：{"mode", "csv_paths", "labels", "metrics", "stats", "curve_fig"}
    """
    paths = expand_csv_inputs(csv_paths, recursive=recursive)
    frames = {p: load_results_csv(p) for p in paths}
    targets = resolve_metrics(metrics, csv_paths=paths)
    names = _dedupe(
        [str(x) for x in labels]
        if labels is not None
        else [read_run_label(p, root) for p in paths]
    )
    single = len(paths) == 1

    print("=" * 70)
    print(
        f"[compare_metrics] 对比模式：{'单文件 · 训练曲线' if single else f'多文件 · 横向对比（{len(paths)} 组）'}"
    )
    for name, path in zip(names, paths):
        epochs = int(frames[path]["epoch"].max())
        print(f"    - {name:<16} epochs={epochs:<4} {path}")
    print(f"[compare_metrics] 对比指标：{', '.join(m.label for m in targets)}")
    print(
        f"[compare_metrics] 统计口径：{'末 ' + str(last_n) + ' epoch' if last_n else '全程（不推荐）'}"
    )
    print("=" * 70)

    out_dir = Path(out_dir).expanduser() if out_dir is not None else None
    stem = out_name or ("metrics_single" if single else "metrics_compare")
    curve_fig = plot_curves(
        paths,
        metrics=targets,
        labels=names,
        title=title,
        last_n=last_n,
        smooth_window=smooth_window,
        std_band=std_band,
        out_path=(out_dir / f"{stem}.png") if out_dir else None,
        dpi=dpi,
        show=show,
    )

    stats: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for metric in targets:
        stats[metric.key] = {
            name: last_n_stats(frame, metric, last_n)
            for name, frame in zip(names, frames.values())
        }
        line = "  ".join(
            f"{name}={stats[metric.key][name]['mean']:.4f}"
            f"(±{stats[metric.key][name]['std']:.4f})"
            for name in names
            if stats[metric.key][name]["mean"] is not None
        )
        print(f"  {metric.label:<22} {line}")

    return {
        "mode": "single" if single else "multi",
        "csv_paths": paths,
        "labels": names,
        "metrics": targets,
        "stats": stats,
        "curve_fig": curve_fig,
    }


if __name__ == "__main__":
    import sys

    # 直接运行本文件时包级导入不可用，退回按路径加载模块自身
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ObjectDetection.compare_metrics import compare_metrics as _compare
    else:
        _compare = compare_metrics

    RUNS = Path(__file__).resolve().parent.parent / "runs/detect/mosaic 10"
    CSV_YOLO11 = RUNS / "baseline/yolo11/results.csv"
    CSV_GHOST = RUNS / "improvement/GhostConv/results.csv"
    CSV_COMBO = RUNS / "improvement/GhostConv_SPPELAN_SimAM/results.csv"

    # 用例 1：先罗列可选指标，凭编号或 key 挑
    list_metrics(CSV_YOLO11)

    # 用例 2：单个 csv -> 单次训练曲线（多指标自动排成子图网格）
    _compare(CSV_YOLO11, metrics=["mAP50-95", "mAP50", "box_loss"], last_n=10)

    # 用例 3：单个 csv + 不指定指标 -> 该文件全部可用指标
    _compare(CSV_YOLO11, metrics=None, last_n=10)

    # 用例 4：多个 csv -> 叠加对比曲线（不画柱状图）
    _compare(
        [CSV_YOLO11, CSV_GHOST, CSV_COMBO],
        metrics=["mAP50-95", "mAP50"],
        labels=["Baseline", "GhostConv", "GhostConv+SPPELAN_SimAM"],
        last_n=10,
    )

    # 用例 5：整目录递归对比全部基线，图例名用目录名
    _compare(RUNS / "baseline", metrics=["mAP50-95", "mAP50"], last_n=10)

    # 用例 6：单指标 + 末 10 epoch 均值参考线 + 曲线平滑，保存到 figures/
    _compare(
        [CSV_YOLO11, CSV_COMBO],
        metrics="mAP50-95",
        labels=["Baseline", "Ours"],
        std_band=True,
        smooth_window=3,
        last_n=10,
        out_dir="figures",
        out_name="mAP_compare",
    )
