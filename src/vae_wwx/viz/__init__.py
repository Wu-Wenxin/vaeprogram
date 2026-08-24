"""matplotlib による最小限の可視化．
对外提供项目绘图函数。"""

from vae_wwx.viz.curves import (
    plot_loss_curves,
    plot_weight_schedule,
)
from vae_wwx.viz.latent import plot_latent_scatter
from vae_wwx.viz.recon import (
    plot_reconstruction_1d,
    plot_reconstruction_channels_overlay,
)


__all__ = [
    "plot_latent_scatter",
    "plot_loss_curves",
    "plot_reconstruction_1d",
    "plot_reconstruction_channels_overlay",
    "plot_weight_schedule",
]