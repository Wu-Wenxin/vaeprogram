"""读取带标签的真实 EMG 数据，并转换成模型训练需要的数据集。

主要流程：
1. 寻找所有 *_labeled.csv 文件。
2. 读取前8列 EMG 信号和第9列标签。
3. 保留跨文件的原始幅值，按 window_size 和 stride 切分窗口。
4. 训练/验证划分后，用训练窗口拟合逐通道标准化参数。
5. 将窗口转换成 (通道数, 窗口长度)。
6. 记录每个窗口来自哪个原始文件。
7. 返回模型训练需要的 {"x", "y"}。

本文件负责读取数据、生成窗口，并提供训练集标准化方法；
训练/验证划分由 Trainer 负责。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 只读取已经完成标签处理的 CSV。
LABELED_CSV_PATTERN = "*_labeled.csv"


class RealEMGDataset(Dataset):
    """读取真实 EMG labeled CSV，并切成 VAE 训练窗口。"""

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int,
        stride: int,
        n_channels: int = 8,
        n_classes: int | None = None,
        seed: int = 0,
    ) -> None:
        # 保存外部传入的数据配置。
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.n_channels = n_channels
        self.config_n_classes = n_classes
        self.seed = seed

        # 递归寻找所有 labeled CSV。
        file_paths = sorted(
            self.data_dir.rglob(LABELED_CSV_PATTERN)
        )
        
        # 没有找到数据时直接报错。
        if not file_paths:
            raise FileNotFoundError(
                f"在 {self.data_dir.resolve()} 下没有找到 "
                f"{LABELED_CSV_PATTERN}"
            )

        # 保存每个文件所属的trial组。
        self.group_strata = [
            _stratum_from_name(path.name)
            for path in file_paths
        ]
        
        # 分别收集所有窗口、标签和来源文件编号。
        all_x: list[np.ndarray] = []
        all_y: list[int] = []
        all_emg_rms: list[float] = []
        all_group_ids: list[int] = []

        # 保存每个 group_id 对应的文件名。
        self.group_names = [
            path.name
            for path in file_paths
        ]

        # 每个 CSV 文件作为一个独立的 group
        for group_id, file_path in enumerate(file_paths):
            # 读取 CSV
            arr = np.loadtxt(
                file_path,
                delimiter=",",
                dtype=np.float32,
            )

            # arr.ndim 表示数组有几个维度；None只是增加一个新的维度
            # 如果只有1个维度，则变为2维
            if arr.ndim == 1:
                arr = arr[None, :]

            # CSV 至少需要包含 EMG 通道和一列标签。
            if arr.shape[1] < self.n_channels + 1:
                raise ValueError(
                    f"{file_path} 列数不足，"
                    f"期望至少 {self.n_channels + 1} 列，"
                    f"实际 {arr.shape[1]} 列"
                )

            # 把EMG信号和标签分开
            raw_signal = arr[:, :self.n_channels]
            # astype(np.int64)转换成整数
            labels = arr[:, self.n_channels].astype(np.int64)

            # T 是当前文件包含的采样点数量。
            t = raw_signal.shape[0]

            # 按照 window_size 和 stride 切分窗口。
            for start in range(
                0,
                t - self.window_size + 1,
                self.stride,
            ):
                end = start + self.window_size

                # 取出当前 EMG 窗口。
                raw_signal_window = raw_signal[start:end]

                # 取出同一时间范围内的标签。
                label_window = labels[start:end]

                # 窗口中存在负标签时跳过。
                if np.any(label_window < 0):
                    continue

                # 窗口中存在超过类别范围的标签时跳过。
                if (
                    self.config_n_classes is not None
                    and np.any(
                        label_window >= self.config_n_classes
                    )
                ):
                    continue

                # 从 (window_size, C) 转成模型需要的 (C, window_size)。
                x = raw_signal_window.T.astype(np.float32)

                # 使用窗口中出现次数最多的标签作为整个窗口的标签。
                y = int(
                    np.bincount(
                        label_window,
                        minlength=self.config_n_classes or 0,
                    ).argmax()
                )

                # 保存当前窗口。
                all_x.append(x)
                all_y.append(y)

                # Distance-Activity Loss用に、標準化前の整流・平滑化EMGから
                # チャネル・時間方向のRMS活動量を保存する。
                all_emg_rms.append(
                    float(np.sqrt(np.mean(raw_signal_window ** 2)))
                )

                # 保存当前窗口来自哪个文件。
                all_group_ids.append(group_id)

        # 一个有效窗口都没有生成时直接报错。
        if not all_x:
            raise RuntimeError(
                "没有生成任何窗口样本，"
                "请检查 window_size、stride 和输入文件长度"
            )

        # 将窗口列表组合成 (N, C, T)。
        x_np = np.stack(all_x, axis=0)

        # 将标签列表转换成 (N,)。
        y_np = np.asarray(all_y, dtype=np.int64)

        # 转换成 PyTorch Tensor。
        self.x = torch.from_numpy(x_np)
        self.y = torch.from_numpy(y_np)
        self.emg_rms = torch.tensor(all_emg_rms, dtype=torch.float32)
        self.normalization: dict[str, list[float]] | None = None

        # 保存每个窗口对应的来源文件编号。
        self.group_ids = torch.tensor(
            all_group_ids,
            dtype=torch.long,
        )

        # 如果没有指定类别数量，就根据实际标签推断。
        self.n_classes = (
            self.config_n_classes
            or int(self.y.max().item()) + 1
        )
        # if self.config_n_classes is not None:
        #     self.n_classes = self.config_n_classes
        # else:
        #     self.n_classes = int(self.y.max().item()) + 1

        # 输出读取结果，方便检查数据。
        label_counts = np.bincount(y_np)
        print(
            f"[RealEMGDataset] files={len(file_paths)} "
            f"windows={len(all_x)} "
            f"x_shape={tuple(self.x.shape)} "
            f"classes={label_counts.tolist()}"
        )

    def __len__(self) -> int:
        """返回数据集中的窗口总数。"""
        return self.x.size(0)

    def standardize_from_train(self, train_indices: list[int]) -> dict[str, list[float]]:
        """仅用训练窗口拟合逐通道参数，再统一处理所有窗口。"""
        if self.normalization is not None:
            raise RuntimeError("数据集已经标准化，不能重复拟合")
        if not train_indices:
            raise ValueError("训练窗口为空，无法计算标准化参数")

        # x 的维度是 (窗口, 通道, 时间)；只合并窗口和时间维度。
        train_x = self.x[train_indices].double()
        mean = train_x.mean(dim=(0, 2), keepdim=True)
        std = train_x.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(1e-6)

        # 训练与验证窗口使用完全相同的参数；原始 emg_rms 不变。
        self.x = (self.x - mean.float()) / std.float()
        self.normalization = {
            "mean": mean.flatten().tolist(),
            "std": std.flatten().tolist(),
        }
        return self.normalization

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:
        """返回指定位置的一个训练样本。"""
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "emg_rms": self.emg_rms[idx],
        }

def _stratum_from_name(name: str) -> str:
    """从文件名中提取trial1、trial2等分层名称。"""

    # 例如从trial2class3_labeled.csv中提取trial2。
    match = re.match(r"^(trial\d+)", name)

    # 文件名不符合规则时归入default组。
    return match.group(1) if match else "default"

if __name__ == "__main__":
    # 这部分只用于单独运行该文件，检查读取结果。
    dataset = RealEMGDataset(
        data_dir="data/rectify_smooth/exp0604_hirota",
        window_size=256,
        stride=128,
        n_channels=8,
        n_classes=5,
    )

    # 取出第一个窗口进行检查。
    sample = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(
        f"Sample x shape: {sample['x'].shape}, "
        f"y: {sample['y']}"
    )
