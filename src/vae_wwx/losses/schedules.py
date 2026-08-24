"""損失重みのスケジューラ．
schedules.py：决定每一步损失权重是多少
        ↓
composite.py：按照权重组合多个损失
すべて ``step (int) -> weight (float)`` の callable オブジェクト．
``CompositeLoss`` の ``schedules`` 辞書に登録して使う．
"""

from __future__ import annotations

from dataclasses import dataclass

from vae_wwx.config.schema import AnnealConfig, KLConfig

class Schedule:
    """重みスケジュールの基底.
    所有权重调度器的共同规则。"""
    # __call__() 让对象可以像函数一样使用schedule(step)
    def __call__(self, step: int) -> float:  # pragma: no cover - abstract
        # 基础类不提供具体算法，子类必须自己实现。
        raise NotImplementedError


@dataclass
class ConstantSchedule(Schedule):
    """
    这个是固定权重的
    不管训练到哪一步，都返回固定权重
    常に ``value`` を返す．"""

    value: float

    def __call__(self, step: int) -> float:  # noqa: ARG002
        return float(self.value)
    
@dataclass
class LinearAnneal(Schedule):
    """0 → ``end_value`` まで ``end_step`` ステップで線形補間．以降は ``end_value`` 固定．
    让权重从0线性增加到最终值。"""

    end_value: float
    end_step: int

    def __call__(self, step: int) -> float:
        # 不需要渐增时，直接返回最终权重。
        if self.end_step <= 0:
            return float(self.end_value)
        # 计算当前训练进度，并限制在0到1之间。
        ratio = min(max(step / self.end_step, 0.0), 1.0)
        return float(self.end_value * ratio)
    
    
def build_schedule(cfg: AnnealConfig, base_weight: float) -> Schedule:
    """``AnnealConfig`` 単独からスケジュールを組み立てる（KL warmup は別関数で合成）．
    根据配置创建权重调度器。"""
    if cfg.kind == "constant":
        return ConstantSchedule(value=base_weight)
    if cfg.kind == "linear":
        return LinearAnneal(end_value=base_weight, end_step=cfg.end_step)
    raise ValueError(f"未対応の anneal kind: {cfg.kind}")


def build_kl_schedule(kl_cfg: KLConfig) -> Schedule:
    """KL の重みスケジュールを ``warmup × anneal`` の合成として返す.
    创建KL损失的权重调度器。
    実装上は両方の値を毎ステップ計算して min を取らず，**warmup と anneal を
    別 schedule で組み合わせて時間方向の積を取る**: ``warmup(step) * (anneal(step) / base)``
    のように書くと冗長なので，ここでは「warmup_steps を超えていれば anneal 通り，
    超えていなければ warmup の比率を掛ける」という単純合成を採用．
    """
    # 创建constant或linear调度器。
    base_anneal = build_schedule(kl_cfg.anneal, base_weight=kl_cfg.weight)
    # 没有warmup时，直接使用基础调度器。
    if kl_cfg.warmup_steps <= 0:
        return base_anneal

    warmup_steps = kl_cfg.warmup_steps

    class _Combined(Schedule):
        """把warmup和基础调度器组合起来。"""
        def __call__(self, step: int) -> float:
            anneal_value = base_anneal(step)
            if step >= warmup_steps:
                return anneal_value
            # warmup 中は線形比率を掛ける
            return float(anneal_value * step / warmup_steps)

    return _Combined()
