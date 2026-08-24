"""``Trainer``: 学習・検証・保存・最終可視化の薄いラッパー．
管理模型训练、验证、数据划分和结果保存。
Lightning 等のフレームワークには依存せず標準 PyTorch だけで完結．
拡張は継承よりも ``train_epoch`` / ``eval_epoch`` の差し替えを想定．
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from vae_wwx.config.schema import ExperimentConfig
from vae_wwx.train.checkpoint import save_checkpoint
from vae_wwx.train.loops import eval_epoch, train_epoch
from vae_wwx.utils.logging_setup import setup_logging
from vae_wwx.utils.seed import set_global_seed


@dataclass
class History:
    """学習履歴（エポック単位の指標と重み推移）．
    保存每个epoch的训练和验证结果。"""
    # 每个epoch的训练损失
    train: list[dict[str, float]] = field(default_factory=list)
    # 每个epoch的验证损失
    val: list[dict[str, float]] = field(default_factory=list)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class Trainer:
    """汎用 Trainer．
    管理一次完整的模型训练。
    Args:
        model: ``forward(x, **cond)`` を持つ ``nn.Module``．
        loss: ``CompositeLoss`` 互換オブジェクト．
        dataset: 1つの ``Dataset``（自動で train/val に分割）．
        cfg: ``ExperimentConfig``．
        cond_keys: モデルに条件として渡すバッチキー（例 ``("y",)`` ）．
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[dict, dict, int], tuple[Tensor, dict, dict]],
        dataset: Dataset,
        cfg: ExperimentConfig,
        cond_keys: tuple[str, ...] = (),
    ) -> None:
        # 固定随机种子，让训练结果尽量可以重复。
        set_global_seed(cfg.train.seed)

        # 保存外部传入的模型、损失、配置和数据集。
        self.model = model
        self.loss = loss
        self.cfg = cfg
        self.cond_keys = cond_keys
        # 根据配置选择训练设备。
        self.device = _resolve_device(cfg.train.device)
        # 创建训练结果保存目录。
        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 创建日志记录器，并把日志写入train.log。
        self.logger: logging.Logger = setup_logging(log_file=self.output_dir / "train.log")
        # 创建专门用于数据划分的随机数生成器。
        gen = torch.Generator().manual_seed(cfg.train.seed)
        # 按完整CSV文件划分训练集和验证集。
        if cfg.data.split_by in ("file","file_stratified",):
            self.train_set, self.val_set = self._split_by_file(dataset, gen)
        # 按单个窗口随机划分训练集和验证集。
        else:
            # 数据集的窗口总数。
            n = len(dataset)
            # 根据验证集比例计算验证窗口数量。
            n_val = max(int(n * cfg.data.val_ratio), 1)
            # 剩余窗口作为训练集。
            n_train = n - n_val
            # 随机划分全部窗口。
            self.train_set, self.val_set = random_split(
                dataset, [n_train, n_val], generator=gen
            )
        # 创建训练DataLoader，并在每个epoch打乱训练数据。
        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.data.batch_size, shuffle=True, drop_last=False
        )
        # 创建验证DataLoader，验证时不打乱数据。
        self.val_loader = DataLoader(
            self.val_set, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False
        )

        # オプティマイザ。创建Adam优化器。
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )
        # 把模型移动到指定设备。
        self.model.to(self.device)
         # 创建空的训练历史。
        self.history = History()
        # 保存训练开始以来的总批次数。
        self.global_step = 0

    def _split_by_file(
        self,
        dataset: Dataset,
        generator: torch.Generator,
    ) -> tuple[Subset, Subset]:
        """按照来源CSV文件划分训练集和验证集。"""
        # 按文件划分要求Dataset提供group_ids。
        if not hasattr(dataset, "group_ids"):
            raise ValueError("data.split_by=file要求Dataset包含group_ids")
        # 取得每个窗口对应的来源文件编号。
        group_ids = torch.as_tensor(getattr(dataset, "group_ids"), dtype=torch.long)
        # 取得所有不重复的文件编号。
        groups = torch.unique(group_ids, sorted=True)
        # 统计原始CSV文件数量。
        n_groups = int(groups.numel())
        # 至少需要两个文件才能划分训练集和验证集。
        if n_groups < 2:
            raise ValueError("data.split_by=file requires at least two source files")
        # file_stratified按照trial分层选择验证文件。
        if self.cfg.data.split_by == "file_stratified":
            val_groups = self._stratified_val_groups(dataset, groups, generator)
        # file直接从全部文件中随机选择验证文件。
        else:
            n_val_groups = max(int(n_groups * self.cfg.data.val_ratio), 1)
            # 至少保留一个文件用于训练。
            n_val_groups = min(n_val_groups, n_groups - 1)
            # 随机打乱所有文件编号。
            perm = torch.randperm(n_groups, generator=generator)
            # 选择验证文件编号。
            val_groups = set(groups[perm[:n_val_groups]].tolist())
        # 收集训练文件产生的窗口索引。
        train_indices = [
            idx for idx, group_id in enumerate(group_ids.tolist()) if group_id not in val_groups
        ]
        # 收集验证文件产生的窗口索引。
        val_indices = [
            idx for idx, group_id in enumerate(group_ids.tolist()) if group_id in val_groups
        ]
        # 取得文件编号对应的CSV文件名。
        group_names = getattr(dataset, "group_names", None)
        # 存在文件名时，把具体划分结果写入日志。
        if group_names is not None:
            val_names = [group_names[group_id] for group_id in sorted(val_groups)]
            train_names = [
                group_names[group_id]
                for group_id in groups.tolist()
                if group_id not in val_groups
            ]
            self.logger.info(f"file split train={train_names} val={val_names}")
        # 记录训练和验证窗口数量。
        self.logger.info(
            f"file split windows train={len(train_indices)} val={len(val_indices)} "
            f"val_ratio={self.cfg.data.val_ratio}"
        )
        # 根据窗口索引创建训练集和验证集。
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def _stratified_val_groups(
        self,
        dataset: Dataset,
        groups: Tensor,
        generator: torch.Generator,
    ) -> set[int]:
        """按照trial分层选择验证CSV文件。"""
        # 取得每个文件对应的trial名称。
        group_strata = getattr(dataset, "group_strata", None)
        # Dataset没有分层信息时无法继续。
        if group_strata is None:
            raise ValueError("data.split_by=file_stratified requires group_strata")
        # 按照trial整理文件编号。
        strata_to_groups: dict[str, list[int]] = {}
        for group_id in groups.tolist():
            strata_to_groups.setdefault(group_strata[group_id], []).append(group_id)
        # 统计全部CSV文件数量。
        n_groups = int(groups.numel())
        # 计算计划使用的验证文件总数。
        n_val_total = max(int(n_groups * self.cfg.data.val_ratio), 1)
        # 至少保留一个文件用于训练。
        n_val_total = min(n_val_total, n_groups - 1)
        # 计算每个trial选择的验证文件数量。
        n_val_per_stratum = max(1, n_val_total // len(strata_to_groups))
        # 收集最终选择的验证文件编号。
        val_groups: set[int] = set()
        # 依次处理trial1、trial2等分层。
        for stratum in sorted(strata_to_groups):
            stratum_groups = strata_to_groups[stratum]
            # 当前trial只有一个文件时不能同时分到两边。
            if len(stratum_groups) < 2:
                continue
            # 至少为当前trial保留一个训练文件。
            n_val = min(n_val_per_stratum, len(stratum_groups) - 1)
            # 随机打乱当前trial中的文件。
            perm = torch.randperm(len(stratum_groups), generator=generator)
            # 加入当前trial选出的验证文件。
            val_groups.update(stratum_groups[i] for i in perm[:n_val].tolist())
        # 一个验证文件都没有选出来时直接报错。
        if not val_groups:
            raise ValueError("data.split_by=file_stratified could not choose validation files")
        return val_groups

    def fit(self, start_epoch: int = 0) -> History:
        """学習を実行．エポック毎に train/val 指標をログし，最終的にチェックポイント保存．
        执行完整训练，并返回训练历史。"""
        # 再次固定训练随机种子。
        set_global_seed(self.cfg.train.seed)
        # 使用较短的变量名读取配置。
        cfg = self.cfg
        # 从start_epoch的下一轮训练到目标epochs。
        for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
            # 训练一个epoch。
            train_metrics, self.global_step = train_epoch(
                model=self.model,
                loss_fn=self.loss,
                optimizer=self.optimizer,
                loader=self.train_loader,
                device=self.device,
                global_step=self.global_step,
                grad_clip=cfg.train.grad_clip,
            )
            # 使用验证集评估模型。
            val_metrics = eval_epoch(
                model=self.model,
                loss_fn=self.loss,
                loader=self.val_loader,
                device=self.device,
                global_step=self.global_step,
                cond_keys=self.cond_keys,
            )
            # 保存当前epoch的训练结果。
            self.history.train.append({"epoch": epoch, **train_metrics})
            # 保存当前epoch的验证结果。
            self.history.val.append({"epoch": epoch, **val_metrics})
            # 将当前结果写入日志。
            self._log_epoch(epoch, train_metrics, val_metrics)

        # 最終チェックポイント
        # 训练结束后保存模型和优化器。
        save_checkpoint(
            self.output_dir / "model.ckpt",
            self.model,
            self.optimizer,
            self.cfg,
        )
        return self.history

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """使用验证集单独评估当前模型。"""
        return eval_epoch(
            model=self.model,
            loss_fn=self.loss,
            loader=self.val_loader,
            device=self.device,
            global_step=self.global_step,
            cond_keys=self.cond_keys,
        )

    def _log_epoch(self, epoch: int, train: dict[str, float], val: dict[str, float]) -> None:
        """将当前epoch的损失写入日志。"""
        # 按固定顺序显示主要损失。
        priority = ["total", "recon", "kl"]
        # 取得当前实际存在的训练指标。
        train_keys = [k for k in priority if k in train]
        # 补充其他可能存在的训练指标。
        train_keys += sorted(k for k in train if k not in train_keys and k != "epoch")
        # 取得当前实际存在的验证指标。
        val_keys = [k for k in priority if k in val]
        # 补充其他可能存在的验证指标。
        val_keys += sorted(k for k in val if k not in val_keys and k != "epoch")
        # 将训练指标转换成字符串。
        train_str = " ".join(f"{k}={train[k]:.4f}" for k in train_keys)
        # 将验证指标转换成字符串。
        val_str = " ".join(f"val_{k}={val[k]:.4f}" for k in val_keys)
        # 写入日志并显示在终端。
        self.logger.info(f"epoch {epoch:3d} | {train_str} | {val_str}")

    @torch.no_grad()
    def get_sample_batch(self, n: int = 8) -> dict[str, Any]:
        """検証セットから ``n`` サンプルだけ取り出す（可視化用）．
        从验证集中取出少量样本用于画图。"""
        # 创建只读取验证样本的DataLoader。
        loader = DataLoader(self.val_set, batch_size=n, shuffle=False)
        # 取得第一个batch。
        batch = next(iter(loader))
        # 将Tensor移动到模型所在设备。
        return {k: (v.to(self.device) if isinstance(v, Tensor) else v) for k, v in batch.items()}
