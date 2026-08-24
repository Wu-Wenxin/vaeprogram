"""YAML → dataclass 再帰変換と CLI 上書き．
读取YAML，并转换成schema.py定义的配置对象。
学習スクリプトでは ``load_config(path)`` で YAML を ``ExperimentConfig`` に
変換します．argparse からの ``--override key=value`` はドット記法で
ネストを辿って上書きします．
yaml 文件
  ↓
loader.py 读取
  ↓
普通 dict
  ↓
schema.py 规定格式
  ↓
ExperimentConfig 对象
  ↓
后面交给 model / data / train 使用
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml

from vae_wwx.config.schema import ExperimentConfig

# Python 3.10+ では ``X | Y`` の origin が ``types.UnionType`` になり，
# ``Union[...]`` の origin (``typing.Union``) と別物になる．両方を受け付ける．
# 同时支持 int | None 和 Union[int, None] 两种写法。
_UNION_ORIGINS = {Union, types.UnionType}


def load_config(path: str | Path) -> ExperimentConfig:
    """YAML ファイルを読み込んで ``ExperimentConfig`` を返す.
    读取YAML配置文件并返回ExperimentConfig。"""
    # 读取YAML，得到普通Python字典。
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    # 将字典递归转换成schema.py中的配置类。
    return _from_dict(ExperimentConfig, raw)


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """``["model.latent_dim=16", "train.epochs=50"]`` 形式の上書きを適用.
    使用命令行参数临时覆盖配置。
    値は YAML として再パースするので ``[1, 2, 3]`` や ``true`` も自然に書けます．
    """
    # 没有覆盖参数时，直接返回原配置。
    if not overrides:
        return cfg
    # 先把配置对象转换成普通字典。
    data = _to_dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override は key=value 形式で指定してください: {item}")
        # 只按照第一个等号拆分。
        key, raw_value = item.split("=", 1)
        # 将字符串重新解析成数字、列表或布尔值。
        value = yaml.safe_load(raw_value)
        # 根据点号找到需要修改的位置。
        _set_nested(data, key.split("."), value)
    # 修改后重新转换成ExperimentConfig。
    return _from_dict(ExperimentConfig, data)

# 递归地将配置对象转换成普通字典。
def _to_dict(obj: Any) -> Any:
    # 如果是dataclass，就把每个字段转换成字典内容。
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    # 如果是列表或元组，逐个转换里面的内容。
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    # 如果本身就是字典，继续转换每个值。
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    # 数字、字符串和None直接返回。
    return obj

# 根据多层键名修改字典中的配置值。
def _set_nested(data: dict, keys: list[str], value: Any) -> None:
    # cur表示当前正在处理的字典位置。
    cur = data
    # 先进入最后一个键之前的所有层级。
    for k in keys[:-1]:
        # 中间层不存在或为None时，自动创建字典。
        if k not in cur or cur[k] is None:
            cur[k] = {}
        # 进入下一层字典。
        cur = cur[k]
    # 修改最后一个键对应的值。
    cur[keys[-1]] = value


def _from_dict(cls: type, data: Any) -> Any:
    """dict → dataclass を型ヒントに従って再帰的に変換.
    根据类型注解，将字典递归转换成dataclass。"""
    # YAML中的null直接转换成None。
    if data is None:
        return None
    # 目标不是dataclass时，进行普通类型转换。
    if not dataclasses.is_dataclass(cls):
        return _coerce(cls, data)
    # dataclass必须使用字典数据创建。
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} には dict が必要ですが {type(data).__name__} を受け取りました")
    # PEP 563 (from __future__ import annotations) のもとでは f.type が str になるので
    # get_type_hints で実型に解決する．
    # 取得类中每个字段的真实类型。
    hints = typing.get_type_hints(cls)
    # 取得这个dataclass允许使用的字段名。
    known_names = {f.name for f in dataclasses.fields(cls)}
    # YAML / override の typo を黙って捨てない．`train.epoch=1` のようなミスを
    # 設定読み込みの時点で必ず失敗させる．
    # YAML中出现拼写错误时，在读取配置阶段直接报错。
    unknown = set(data.keys()) - known_names
    if unknown:
        raise ValueError(
            f"{cls.__name__} に未知のキーが含まれています: {sorted(unknown)} "
            f"(有効なキー: {sorted(known_names)})"
        )
    # 收集创建dataclass需要的参数。
    kwargs: dict[str, Any] = {}
    # 依次处理dataclass中的每个字段。
    for f in dataclasses.fields(cls):
        # YAML没有填写时，保留schema.py中的默认值。
        if f.name not in data:
            continue
        # 取得当前字段的真实类型。
        ftype = hints.get(f.name, f.type)
        # 递归转换当前字段的数据。
        kwargs[f.name] = _from_dict(_resolve_type(ftype), data[f.name])
    # 使用转换后的参数创建配置对象。
    return cls(**kwargs)


def _resolve_type(tp: Any) -> Any:
    """``int | None`` のような Optional をはがして実型を返す（再帰用途）.
    从可选类型中取出真正需要转换的类型。"""
    # 取得类型的外层结构。
    origin = get_origin(tp)
    # 判断是不是int | None等联合类型。
    if origin in _UNION_ORIGINS:
        # 去掉None，只保留真正的数据类型。
        args = [a for a in get_args(tp) if a is not type(None)]
        # 只有一个非None类型时，直接返回它。
        if len(args) == 1:
            return args[0]
    # 不是可选类型时，保持原样。
    return tp


def _coerce(tp: Any, value: Any) -> Any:
    """非 dataclass 型に対する最小限の coerce（list / tuple / プリミティブ）.
    转换列表、元组和普通数据类型。"""
    # 取得list、tuple等外层类型。
    origin = get_origin(tp)
    # 处理list[int]等列表类型。
    if origin in (list,):
        # 取得列表内部元素的类型。
        (inner,) = get_args(tp) or (Any,)
        # 逐个转换列表中的元素。
        return [_coerce(inner, v) for v in value]
    # 处理tuple[float, float]等元组类型。
    if origin in (tuple,):
        inners = get_args(tp)
        # 处理tuple[int, ...]这种不固定长度的元组。
        if len(inners) == 2 and inners[1] is Ellipsis:
            return tuple(_coerce(inners[0], v) for v in value)
        # 处理tuple[float, float]这种固定长度的元组。
        return tuple(_coerce(t, v) for t, v in zip(inners, value, strict=False))
    # 处理int | None等联合类型。
    if origin in _UNION_ORIGINS:
        # Optional 等．None なら None，そうでなければ非 None 型で再帰．
        # YAML中的null直接转换成None。
        if value is None:
            return None
        # 尝试使用每一种非None类型进行转换。
        for a in get_args(tp):
            if a is type(None):
                continue
            try:
                return _coerce(a, value)
            except (TypeError, ValueError):
                continue
        # 都无法转换时，保留原值。
        return value
    # 如果目标类型还是dataclass，继续递归转换。
    if dataclasses.is_dataclass(tp):
        return _from_dict(tp, value)
    # 字符串、数字和Literal等保持YAML解析后的值。
    return value
