import random
from ultralytics import YOLO, YOLOWorld


def train(
    model_selection: str,
    yaml_data: str,
    yolo_world: bool = False,
    epochs: int = 100,
    batch: int = -1,
    val: bool = True,
    save_period: int = -1,
    project: str = None,
    pretrained: str = None,
    single_cls: bool = False,
    lr: float = 0.001,
    workers: int = 0,
    seed_change: bool = False,
    cls: float = 0.5,
    imgsz: int = 640,
    optimizer="Adam",
    patience=100,
    resume: bool = False,
    plots=True,
    cos_lr=True,
    iou: float = 0.7,
):
    # seed_change=True 时每轮实验用不同随机种子（做多次重复实验取均值）；
    # False 时固定 seed=1，保证同一份配置跑出同样结果，便于复现和调参对比。
    seed = random.randint(1, 1e9) if seed_change else 1

    # YOLOWorld 是开放词表模型，加载方式与普通 YOLO 不同，需分支处理
    if yolo_world:
        model = YOLOWorld(model_selection)
    else:
        model = YOLO(model_selection)

    model.train(
        data=yaml_data,  # 数据集配置 yaml
        epochs=epochs,  # 总训练轮数
        workers=workers,  # dataloader 线程数
        batch=batch,  # 批次大小，-1 为自动
        save_period=save_period,  # 周期性存档的间隔
        val=val,  # 是否每轮结束后做验证
        pretrained=pretrained,  # 预训练权重
        project=project,  # 输出目录
        lr0=lr,  # 初始学习率
        single_cls=single_cls,  # 是否合并为单类
        imgsz=imgsz,  # 输入分辨率
        seed=seed,  # 随机种子
        cls=cls,  # 分类损失增益
        optimizer=optimizer,  # 优化器
        patience=patience,  # 早停轮数
        resume=resume,  # 断点续训
        plots=plots,  # 是否画结果图
        cos_lr=cos_lr,  # 余弦退火学习率
        iou=iou,  # NMS IoU 阈值
    )
