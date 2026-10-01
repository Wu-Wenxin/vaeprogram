"""データセットファクトリ．
根据数据配置创建对应的数据集。
config の ``DataConfig.name`` に応じて適切な ``Dataset`` を返します．
段階的に対応データ種を増やしていく前提で，未対応の ``name`` は
``ValueError`` を出して落とします．
"""

from __future__ import annotations

from torch.utils.data import Dataset

from vae_wwx.config.schema import DataConfig


def build_dataset(cfg: DataConfig, seed: int | None = None) -> Dataset:
    """``DataConfig`` から学習用データセットを生成."""
    actual_seed = cfg.seed if seed is None else seed
    if cfg.name == "real_emg":
        from vae_wwx.data.read_real_emg import RealEMGDataset

        if cfg.data_dir is None:
            raise ValueError("real_emg では DataConfig.data_dir の指定が必要です")
        return RealEMGDataset(
            data_dir=cfg.data_dir,
            window_size=cfg.length,
            stride=cfg.stride or cfg.length,
            n_channels=cfg.channels,
            n_classes=cfg.n_classes,
            seed=actual_seed,
        )
    raise ValueError(f"未対応のデータ名: {cfg.name}")
