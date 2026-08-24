"""標準 ``logging`` のセットアップ.
设置项目使用的日志记录器。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """ルートロガーを整え，``vae_factory`` ロガーを返す.
    设置终端和文件日志，并返回项目日志记录器。
    複数回呼んでも安全．既存ハンドラはクリアして再設定します．
    """
    # 取得Python的根日志记录器。
    root = logging.getLogger()

    # 删除以前存在的日志处理器，避免重复打印。
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # 设置需要记录的最低日志级别。
    root.setLevel(level)

    # 规定每条日志的显示格式。
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 创建终端日志处理器。
    stream_handler = logging.StreamHandler(sys.stdout)
    # 给终端日志应用统一格式。
    stream_handler.setFormatter(formatter)
    # 将终端处理器加入根日志记录器。
    root.addHandler(stream_handler)
    # 指定日志文件时，同时保存文件日志。
    if log_file is not None:
        # 确保日志文件所在目录存在。
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 创建文件日志处理器。
        file_handler = logging.FileHandler(log_file)
        # 给文件日志应用相同格式。
        file_handler.setFormatter(formatter)
        # 将文件处理器加入根日志记录器。
        root.addHandler(file_handler)
    # 返回当前项目专用的日志记录器。
    return logging.getLogger("vae_wwx")
