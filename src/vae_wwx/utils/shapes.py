"""shape assert ユーティリティ.
检查Tensor的实际形状是否符合预期。
forward 中の shape を「期待形と比較して通すだけ」のヘルパー．
``-1`` は任意の長さ，``"B"`` のような文字列は同名同士で同値性のみチェック．
"""

from __future__ import annotations

from torch import Tensor


def assert_shape(tensor: Tensor, expected: tuple, name: str = "tensor") -> None:
    """期待 shape と一致するかチェック．不一致なら ``AssertionError``.
    检查Tensor形状，不符合时抛出AssertionError。
    例:
        ``assert_shape(x, ("B", 4, 128), name="EMG入力")``

    ``int`` は数値一致，``str`` は同じ ``str`` 同士の同値性，``-1`` は任意一致．
    """
    # 取得Tensor的实际形状。
    actual = tuple(tensor.shape)
    # 首先检查维度数量是否一致。
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: 次元数が違います．期待 {expected}，実際 {actual}")
    # 保存B、C、T等命名维度对应的实际数值。
    bindings: dict[str, int] = {}
    # 逐个比较每一个维度。i是维度位置，e是期望大小，a是实际大小。
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        # 整数表示要求固定大小。
        if isinstance(e, int):
            # -1表示当前维度可以是任意大小。数值相同表示检查通过。
            if e == -1 or e == a:
                continue
            raise AssertionError(f"{name}[{i}]: 期待 {e}，実際 {a}（全体: 期待 {expected} 実際 {actual}）")
        # 字符串表示命名维度。
        if isinstance(e, str):
            # 同名维度已经出现时，检查数值是否一致。
            if e in bindings:
                if bindings[e] != a:
                    raise AssertionError(
                        f"{name}[{i}]: 同名軸 '{e}' で値が一致しません（{bindings[e]} vs {a}）"
                    )
            # 第一次出现时保存对应的实际大小。
            else:
                bindings[e] = a
            continue
        raise TypeError(f"shape の要素は int か str だけにしてください（{e}）")
