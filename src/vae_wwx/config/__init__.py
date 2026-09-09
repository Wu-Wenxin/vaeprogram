"""对外提供配置类和配置读取函数。"""

from vae_wwx.config.loader import (
    apply_overrides,
    load_config,
)
from vae_wwx.config.schema import (
    AnnealConfig,
    DataConfig,
    DistanceActivityConfig,
    ExperimentConfig,
    KLConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    ReconConfig,
    TrainConfig,
    VizConfig,
)

__all__ = [
    "AnnealConfig",
    "DataConfig",
    "DistanceActivityConfig",
    "ExperimentConfig",
    "KLConfig",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "ReconConfig",
    "TrainConfig",
    "VizConfig",
    "apply_overrides",
    "load_config",
]
