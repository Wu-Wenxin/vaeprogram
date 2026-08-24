import torch
from vae_wwx.models.decoders_1d import SimpleConv1dDecoder
from vae_wwx.models.encoders_1d import SimpleConv1dEncoder
from vae_wwx.models.vae_vanilla import VanillaVAE

from vae_wwx.losses.kl import gaussian_kl
from vae_wwx.losses.reconstruction import mse_recon_loss

def main() -> None:
    batch_size = 4
    channels = 8
    length = 128
    latent_dim = 8
    hidden_channels = [16, 32 ,64]
    
    # 生成假数据：(样本数, 通道数, 时间长度)
    x = torch.randn(
        batch_size,
        channels,
        length,
    )

    encoder = SimpleConv1dEncoder(
        in_channels=channels,
        hidden_channels=hidden_channels,
        length=length,
    )

    decoder = SimpleConv1dDecoder(
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
        out_channels=channels,
        length=length,
        bottleneck_length=encoder.bottleneck_length,
    )

    model = VanillaVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(x)

    print("x shape:", x.shape)
    print("x_hat shape:", outputs["x_hat"].shape)
    print("mu shape:", outputs["mu"].shape)
    print("logvar shape:", outputs["logvar"].shape)
    print("z shape:", outputs["z"].shape)

    assert outputs["x_hat"].shape == x.shape
    assert outputs["mu"].shape == (batch_size, latent_dim)
    assert outputs["logvar"].shape == (batch_size, latent_dim)
    assert outputs["z"].shape == (batch_size, latent_dim)
    print("模型形状检查通过。")

    reconstruction_loss = mse_recon_loss(
        outputs["x_hat"],
        x,
    )

    kl_loss = gaussian_kl(
        outputs["mu"],
        outputs["logvar"],
    )

    total_loss = reconstruction_loss + kl_loss

    print("reconstruction loss:", reconstruction_loss.item())
    print("KL loss:", kl_loss.item())
    print("total loss:", total_loss.item())

    assert torch.isfinite(reconstruction_loss)
    assert torch.isfinite(kl_loss)
    assert torch.isfinite(total_loss)

    print("损失函数检查通过。")

if __name__ == "__main__":
    main()