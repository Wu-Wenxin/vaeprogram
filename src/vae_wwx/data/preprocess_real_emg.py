"""预处理真实 EMG，并保存处理前后的8通道对比图。

功能：
1. 读取单个 CSV 或遍历整个文件夹。
2. 取原始数据的前8个 EMG 通道。
3. 调用整流平滑函数。
4. 保存 *_filtered.csv。
5. 保存原始信号与处理后信号的对比图。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

# 使用无窗口绘图模式，方便在终端运行。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vae_wwx.data import preprocess_rectify_smooth


# 不提供命令行路径时使用这个默认路径。
DEFAULT_INPUT_PATH = Path("data/real-emg/exp0703_hirota_trial1")

# 所有整流平滑结果保存到这个目录。
OUTPUT_ROOT = Path("data/rectify_smooth")

# EMG 采样频率，用于把横轴转换成秒。
FS = 2000.0

# 原始数据使用前8个 EMG 通道。
N_CHANNELS = 8

# 只匹配 trial1class1.csv 这类原始文件。
RAW_FILE_PATTERN = re.compile(r"^trial\d+class\d+\.csv$")


def plot_comparison(
    raw: np.ndarray,
    filtered: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """绘制8通道原始信号和处理后信号的对比图。"""

    # 根据采样频率生成时间轴，单位为秒。
    time = np.arange(raw.shape[0]) / FS

    # 创建8行2列的图片。
    # 每一行对应一个 EMG 通道。
    # 左边显示原始信号，右边显示处理后信号。
    figure, axes = plt.subplots(
        N_CHANNELS,
        2,
        figsize=(16, 16),
        sharex=True,
        sharey="col",  # 左右两列纵轴分别使用相同的范围
    )

    # 逐通道绘制信号。
    for channel in range(N_CHANNELS):
        # 左侧绘制原始 EMG。
        axes[channel, 0].plot(
            time,
            raw[:, channel],
            linewidth=0.5,
            color="steelblue",
        )

        # 右侧绘制整流平滑后的 EMG。
        axes[channel, 1].plot(
            time,
            filtered[:, channel],
            linewidth=0.5,
            color="darkorange",
        )

        # 显示当前通道编号。
        axes[channel, 0].set_ylabel(f"Ch {channel + 1}")

        # 添加浅色网格，方便观察信号变化。
        axes[channel, 0].grid(alpha=0.2)
        axes[channel, 1].grid(alpha=0.2)

    # 设置左右两列标题。
    axes[0, 0].set_title("Raw EMG")
    axes[0, 1].set_title("Rectified + smoothed EMG")

    # 只在最下面一行显示横轴名称。
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")

    # 设置整张图片的标题。
    figure.suptitle(title)

    # 自动调整子图间距。
    figure.tight_layout()

    # 创建图片输出目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图片。
    figure.savefig(output_path, dpi=120)

    # 关闭图片，避免批量处理时占用过多内存。
    plt.close(figure)


def process_file(file_path: Path) -> Path:
    """处理一个原始 EMG CSV，并返回 filtered CSV 路径。"""

    # 当前文件所在目录名称作为实验名称。
    experiment_name = file_path.parent.name

    # 生成当前实验的输出目录。
    output_dir = OUTPUT_ROOT / experiment_name

    # 图片统一保存在输出目录里面的 pic 文件夹。
    picture_dir = output_dir / "pic"

    # 创建 CSV 输出目录。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建图片输出目录。
    picture_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始 CSV。
    raw = np.loadtxt(
        file_path,
        delimiter=",",
        dtype=np.float32,
    )

    # 保留原项目中的二维数组检查。
    if raw.ndim != 2:
        raise ValueError(
            f"{file_path} expected 2D array (T, C), got {raw.shape}"
        )

    # 保留原项目中的最少8通道检查。
    if raw.shape[1] < N_CHANNELS:
        raise ValueError(
            f"{file_path} expected at least {N_CHANNELS} "
            f"EMG channels, got {raw.shape[1]}"
        )

    # 与原项目一致，只使用前8列。
    raw_emg = raw[:, :N_CHANNELS]

    # 对原始 EMG 进行整流和平滑。
    filtered = preprocess_rectify_smooth(raw_emg)

    # 生成 filtered CSV 路径。
    filtered_path = output_dir / f"{file_path.stem}_filtered.csv"

    # 保存处理后的 EMG。
    np.savetxt(
        filtered_path,
        filtered,
        delimiter=",",
        fmt="%.8e",
    )

    # 生成对比图路径。
    picture_path = picture_dir / f"{file_path.stem}_compare.png"

    # 绘制并保存8通道对比图。
    plot_comparison(
        raw=raw_emg,
        filtered=filtered,
        title=file_path.name,
        output_path=picture_path,
    )

    return filtered_path


def process_directory(input_dir: Path) -> list[Path]:
    """遍历目录并处理所有符合规则的原始 CSV。"""

    # 递归寻找 trial1class1.csv 这类原始文件。
    file_paths = sorted(
        path
        for path in input_dir.rglob("*.csv")
        if RAW_FILE_PATTERN.fullmatch(path.name)
    )

    # 没有找到文件时给出提示。
    if not file_paths:
        print(f"No raw EMG files found under {input_dir}")
        return []

    # 保存所有生成的 filtered CSV 路径。
    filtered_paths: list[Path] = []

    # 逐个处理原始 CSV。
    for file_path in file_paths:
        filtered_path = process_file(file_path)

        # 保存输出路径。
        filtered_paths.append(filtered_path)

        # 显示当前处理进度。
        print(f"[OK] {file_path.name} -> {filtered_path}")

    return filtered_paths


def main() -> None:
    """根据输入路径处理一个 CSV 或整个文件夹。"""

    # 命令后提供路径时使用用户路径。
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

    # 没有提供路径时使用默认路径。
    else:
        input_path = DEFAULT_INPUT_PATH

    # 输入是单个文件时，只处理这个 CSV。
    if input_path.is_file():
        filtered_path = process_file(input_path)
        filtered_paths = [filtered_path]

        print(f"[OK] {input_path.name} -> {filtered_path}")

    # 输入是文件夹时，遍历其中的原始 CSV。
    elif input_path.is_dir():
        filtered_paths = process_directory(input_path)

    # 路径不存在时停止运行。
    else:
        raise FileNotFoundError(
            f"Input path not found: {input_path}"
        )

    # 显示最终处理数量。
    print(f"\nSummary: processed {len(filtered_paths)} files")


if __name__ == "__main__":
    main()