from ultralytics import YOLO


def calc_complexity_yolo(
    model_selection: str, imgsz: int = 640, fuse: bool = True
) -> dict:
    """统计单个 YOLO .pt 权重的复杂度。

    Args:
        model_selection: .pt 权重路径
        imgsz: 输入尺寸（默认 640）
        fuse: 是否先做 Conv+BN 融合（默认 True,与验证 / 部署一致 设为 False 则统计含独立 BN 的未融合模型）
    Returns:
        {"model", "imgsz", "fuse", "params_M", "GFLOPs"}（参数量单位 M,计算量单位 GFLOPs,均保留 3 位小数后四舍五入）
    """

    model = YOLO(model_selection)

    if fuse:
        model.model.fuse()  # BN 折叠进 Conv：参数 / 计算量与验证、部署一致

    n_layers, n_params, n_grads, gflops = model.info(verbose=True, imgsz=imgsz)

    params_m = n_params / 1e6

    print("\n" + "=" * 54)
    print("  YOLO 复杂度（ultralytics 官方口径,FLOPs = 2×MACs）")
    print("=" * 54)
    print(f"  权重     : {model_selection}")
    print(f"  输入尺寸 : {imgsz} × {imgsz}")
    print(f"  fuse     : {fuse}（True=BN折叠,与验证/部署一致）")
    print(f"  层数     : {n_layers:,}")
    print(f"  参数量   : {params_m:.3f} M   ({n_params:,})")
    print(f"  计算量   : {gflops:.3f} GFLOPs")
    print("=" * 54)

    return {
        "model": model_selection,
        "imgsz": imgsz,
        "fuse": fuse,
        "params_M": round(params_m, 3),
        "GFLOPs": round(gflops, 3),
    }
