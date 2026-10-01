"""根据人工修改后的 segments CSV 重新生成标签。

使用流程：
1. 自动标注失败后，查看 pic 目录中的诊断图。
2. 手动修改 *_segments.csv。
3. 运行当前文件。
4. 重新生成 *_labeled.csv 和 pic/*_diag.png。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

# 使用不需要打开窗口的绘图模式。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vae_wwx.data.label_segments import (
    detect_segments,
    diag_plot,
    rms_envelope,
    sampen,
)


# 没有提供路径时使用这个默认目录。
DEFAULT_INPUT_PATH = Path(
    "data/rectify_smooth/exp0604_hirota"
)

# EMG 采样频率。
FS = 2000.0

# 允许 end_sample 与 duration_s 之间相差1个采样点。
TOLERANCE_SAMPLES = 1

# 在目录中寻找这种文件。
SEGMENTS_GLOB = "*_segments.csv"


def read_segments(
    path: Path,
) -> list[tuple[int, int, int, float | None]]:
    """读取 segments CSV。"""

    # 保存读取到的所有动作区间。
    rows: list[
        tuple[int, int, int, float | None]
    ] = []

    # 打开 segments CSV。
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # 根据表头读取每一行。
        reader = csv.DictReader(file)

        # 规定必须存在的列。
        required = {
            "start_sample",
            "end_sample",
            "label",
        }

        # 找出缺少的列。
        missing = required - set(
            reader.fieldnames or []
        )

        # 缺少必要列时停止。
        if missing:
            raise ValueError(
                f"{path.name}: "
                f"missing columns {missing}"
            )

        # 逐行读取动作区间。
        for row in reader:
            # duration_s 可以不存在。
            duration_s = None

            # 存在 duration_s 时转换成浮点数。
            if row.get("duration_s") not in (
                None,
                "",
            ):
                duration_s = float(
                    row["duration_s"]
                )

            # 保存当前动作区间。
            rows.append(
                (
                    int(row["start_sample"]),
                    int(row["end_sample"]),
                    int(row["label"]),
                    duration_s,
                )
            )

    return rows


def normalize_segments(
    segments: list[
        tuple[int, int, int, float | None]
    ],
    n_samples: int,
    source_name: str,
) -> list[tuple[int, int, int]]:
    """检查并修正人工编辑后的动作区间。"""

    # 保存规范化后的动作区间。
    normalized: list[
        tuple[int, int, int]
    ] = []

    # 从CSV第2行开始检查。
    for row_index, (
        start,
        end,
        label,
        duration_s,
    ) in enumerate(segments, start=2):
        # 标签仍为-1时说明没有完成人工修改。
        if label < 0:
            raise ValueError(
                f"{source_name}: row {row_index} "
                f"still has label={label}; "
                f"edit it before applying segments."
            )

        # 限制开始位置不能超出信号范围。
        start = max(
            0,
            min(start, n_samples),
        )

        # 限制结束位置不能超出信号范围。
        end = max(
            0,
            min(end, n_samples),
        )

        # 提供 duration_s 时，根据持续时间检查结束位置。
        if duration_s is not None:
            # 根据开始位置和持续时间计算结束位置。
            expected_end = start + int(
                round(duration_s * FS)
            )

            # 限制计算结果不能超出信号范围。
            expected_end = max(
                0,
                min(expected_end, n_samples),
            )

            # 差异超过允许范围时修正 end_sample。
            if (
                abs(end - expected_end)
                > TOLERANCE_SAMPLES
            ):
                print(
                    f"[FIX] {source_name}: "
                    f"row {row_index} "
                    f"end_sample {end} "
                    f"-> {expected_end} "
                    f"based on "
                    f"duration_s={duration_s:.4f}"
                )

                end = expected_end

        # 开始位置必须小于结束位置。
        if start >= end:
            print(
                f"[SKIP] {source_name}: "
                f"row {row_index} "
                f"has invalid range "
                f"start={start}, end={end}",
                file=sys.stderr,
            )

            continue

        # 保存有效动作区间。
        normalized.append(
            (start, end, label)
        )

    return normalized


def build_labels(
    n_samples: int,
    segments: list[tuple[int, int, int]],
) -> np.ndarray:
    """根据动作区间生成逐采样点标签。"""

    # 默认所有位置都是 rest，标签为0。
    labels = np.zeros(
        n_samples,
        dtype=np.int8,
    )

    # 将每个动作区间设置为指定标签。
    for start, end, label in segments:
        labels[start:end] = label

    return labels


def write_segments_csv(
    path: Path,
    segments: list[tuple[int, int, int]],
) -> None:
    """将修正后的动作区间写回 segments CSV。"""

    # 打开原 segments CSV，并覆盖旧内容。
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        # 创建 CSV 写入器。
        writer = csv.writer(file)

        # 写入表头。
        writer.writerow(
            [
                "start_sample",
                "end_sample",
                "label",
                "duration_s",
            ]
        )

        # 逐行写入规范化后的动作区间。
        for start, end, label in segments:
            # 根据采样点计算持续时间。
            duration_s = (
                end - start
            ) / FS

            writer.writerow(
                [
                    start,
                    end,
                    label,
                    f"{duration_s:.4f}",
                ]
            )


def process_file(
    segments_csv: Path,
) -> bool:
    """处理一个人工修改后的 segments CSV。"""

    # 去掉文件名中的 _segments。
    base = segments_csv.stem.replace(
        "_segments",
        "",
    )

    # 找到对应的 filtered CSV。
    filtered_path = segments_csv.with_name(
        f"{base}_filtered.csv"
    )

    # filtered CSV 不存在时无法重新生成标签。
    if not filtered_path.exists():
        print(
            f"[SKIP] {segments_csv.name}: "
            f"{filtered_path.name} not found",
            file=sys.stderr,
        )

        return False

    # 读取 filtered EMG。
    x = np.loadtxt(
        filtered_path,
        delimiter=",",
    )

    # 一维输入转换成单通道二维形式。
    if x.ndim == 1:
        x = x[:, None]

    # 读取人工修改后的动作区间。
    raw_segments = read_segments(
        segments_csv
    )

    # 检查并修正动作区间。
    segments = normalize_segments(
        raw_segments,
        x.shape[0],
        segments_csv.name,
    )

    # 将规范化后的结果写回 segments CSV。
    write_segments_csv(
        segments_csv,
        segments,
    )

    # 根据动作区间生成逐采样点标签。
    labels = build_labels(
        x.shape[0],
        segments,
    )

    # 将标签添加到 EMG 最后一列。
    labeled = np.hstack(
        [
            x,
            labels.reshape(-1, 1),
        ]
    )

    # 生成 labeled CSV 路径。
    labeled_path = segments_csv.with_name(
        f"{base}_labeled.csv"
    )

    # 保存 labeled CSV。
    np.savetxt(
        labeled_path,
        labeled,
        delimiter=",",
        fmt=(
            ["%.8e"] * x.shape[1]
            + ["%d"]
        ),
    )

    # 诊断图保存在 pic 文件夹。
    diagnostic_path = (
        segments_csv.parent
        / "pic"
        / f"{base}_diag.png"
    )

    # 重新计算 RMS 包络。
    env, centers = rms_envelope(x)

    # 重新计算自动检测阈值，用于画诊断图。
    _, threshold, threshold_label = (
        detect_segments(
            env,
            centers,
            x.shape[0],
        )
    )

    # 找出变化最大的通道。
    most_active_channel = int(
        np.argmax(x.var(axis=0))
    )

    # 按开始位置排序动作区间。
    ordered_segments = sorted(
        (start, end)
        for start, end, _label in segments
    )

    # 重新计算每个动作段的 Sample Entropy。
    sampen_per_segment = [
        sampen(
            x[
                start:end,
                most_active_channel,
            ]
        )
        for start, end in ordered_segments
    ]

    # 生成诊断图标题。
    title = (
        f"{filtered_path.name}  "
        f"thr={threshold_label}  "
        f"applied manual segments"
    )

    # 保存新的诊断图。
    diag_plot(
        env=env,
        centers=centers,
        threshold=threshold,
        segments=ordered_segments,
        sampen_per_seg=sampen_per_segment,
        n_samples=x.shape[0],
        title=title,
        output_path=diagnostic_path,
    )

    # 统计每种标签的采样点数量。
    counts = {
        int(label): int(
            (labels == label).sum()
        )
        for label in np.unique(labels)
    }

    # 显示处理结果。
    print(
        f"[OK] {segments_csv.name} "
        f"-> {labeled_path.name}, "
        f"{diagnostic_path.name} "
        f"labels={counts}"
    )

    return True


def collect_segment_files(
    target: Path,
) -> list[Path]:
    """收集单个 segments 文件或目录中的全部 segments 文件。"""

    # 输入是单个文件时直接返回。
    if target.is_file():
        return [target]

    # 路径不存在时停止。
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found"
        )

    # 输入既不是文件也不是目录时停止。
    if not target.is_dir():
        raise ValueError(
            f"{target} is neither "
            f"a file nor a directory"
        )

    # 递归寻找所有 segments CSV。
    return sorted(
        target.rglob(SEGMENTS_GLOB)
    )


def main(
    target: Path,
) -> None:
    """处理一个 segments CSV 或整个目录。"""

    # 收集需要处理的 segments 文件。
    files = collect_segment_files(target)

    # 没有找到文件时给出提示。
    if not files:
        print(
            f"No {SEGMENTS_GLOB} "
            f"under {target}",
            file=sys.stderr,
        )

        return

    # 记录成功处理数量。
    n_ok = 0

    # 逐个处理 segments CSV。
    for path in files:
        if process_file(path):
            n_ok += 1

    # 显示处理汇总。
    print(
        f"\nSummary: applied "
        f"{n_ok} / {len(files)}"
    )


if __name__ == "__main__":
    # 提供路径时使用命令行路径。
    if len(sys.argv) >= 2:
        target_path = Path(sys.argv[1])

    # 没有提供路径时使用默认目录。
    else:
        target_path = DEFAULT_INPUT_PATH

    # 开始处理。
    main(target_path)