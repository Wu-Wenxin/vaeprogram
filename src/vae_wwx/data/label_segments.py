"""对滤波后的 EMG 自动检测动作段并写入标签。

流程：
1. 计算8通道 EMG 的 RMS envelope。
2. 使用多个候选阈值检测 active 区间。
3. 检查动作数量、动作时长和 rest 间隔。
4. 检测成功时，将 rest 标为0，动作依次标为1到4。
5. 检测失败时，将标签写成-1，等待人工检查。
6. 输出 labeled、segments 和诊断图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# 使用不需要打开窗口的绘图模式。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# 没有提供路径时使用这个默认目录。
DEFAULT_INPUT_PATH = Path(
    "data/rectify_smooth/exp0703_hirota_trial1"
)

# EMG 采样频率。
FS = 2000.0

# RMS 窗口长度：400个采样点，即200 ms。
WIN = 400

# RMS 窗口步长：100个采样点，即50 ms。
HOP = 100

# RMS 包络的移动平均窗口。
SMOOTH_WIN = 5

# 动作允许的最短持续时间。
MIN_ACTIVE_S = 1.5

# 动作允许的最长持续时间。
MAX_ACTIVE_S = 4.5

# 动作之间最短的 rest 时间。
MIN_REST_S = 0.5

# 小于0.3秒的短间隔会被合并。
GAP_CLOSE_S = 0.3

# 每个文件预期包含4个动作。
EXPECTED_ACTIVE = 4

# Otsu 和 MAD 失败时尝试的百分位阈值。
PERCENTILE_THRESHOLDS = (50, 55, 60, 65, 70)


def rms_envelope(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算多通道 EMG 的 RMS envelope。"""

    # 取得总采样点数量。
    n = x.shape[0]

    # 计算每个 RMS 窗口的开始位置。
    starts = np.arange(0, n - WIN + 1, HOP)

    # 保存每个窗口、每个通道的 RMS。
    rms = np.empty((starts.size, x.shape[1]))

    # 逐窗口计算 RMS。
    for i, start in enumerate(starts):
        # 截取当前窗口。
        seg = x[start : start + WIN]

        # 分别计算每个通道的 RMS。
        rms[i] = np.sqrt(np.mean(seg * seg, axis=0))

    # 对所有通道求平均。
    env = rms.mean(axis=1)

    # 对 RMS 包络进行移动平均。
    if SMOOTH_WIN > 1:
        kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
        env = np.convolve(env, kernel, mode="same")

    # 计算每个窗口中心对应的采样点。
    centers = starts + WIN // 2

    return env, centers


def otsu_threshold(
    values: np.ndarray,
    nbins: int = 256,
) -> float:
    """使用 Otsu 方法寻找 rest 和 active 的分割阈值。"""

    # 统计包络值的直方图。
    hist, edges = np.histogram(values, bins=nbins)

    # 将直方图转换成概率。
    prob = hist.astype(float) / max(hist.sum(), 1)

    # 计算每个区间的中心值。
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 计算累计概率。
    omega = np.cumsum(prob)

    # 计算累计均值。
    mu = np.cumsum(prob * centers)

    # 取得整体均值。
    mu_t = mu[-1]

    # 计算类间方差的分母。
    denom = omega * (1 - omega)

    # 避免分母为0。
    denom[denom == 0] = 1e-12

    # 计算类间方差。
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    # 返回类间方差最大位置对应的阈值。
    return float(
        centers[int(np.argmax(sigma_b2))]
    )


def runs_of_true(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """把 True/False mask 转换成连续 True 区间。"""

    # 空 mask 没有区间。
    if mask.size == 0:
        return []

    # 找出 False 和 True 之间的变化位置。
    diff = np.diff(
        mask.astype(np.int8),
        prepend=0,
        append=0,
    )

    # 1 表示 True 区间开始。
    starts = np.where(diff == 1)[0]

    # -1 表示 True 区间结束。
    ends = np.where(diff == -1)[0]

    # 将开始和结束位置组合起来。
    return list(
        zip(starts.tolist(), ends.tolist())
    )


def detect_segments(
    env: np.ndarray,
    centers: np.ndarray,
    n_samples: int,
) -> tuple[list[tuple[int, int]], float, str]:
    """检测 active segments，并返回实际使用的阈值。"""

    # 保存每个候选阈值的检测结果。
    candidates = []

    # 计算 Otsu 阈值。
    thr_otsu = otsu_threshold(env)

    # 计算中位数。
    median = float(np.median(env))

    # 计算 MAD。
    mad = (
        float(np.median(np.abs(env - median)))
        or 1e-12
    )

    # 根据 MAD 计算候选阈值。
    thr_mad = median + 3.0 * 1.4826 * mad

    # 首先尝试 Otsu 和 MAD。
    threshold_candidates = [
        ("otsu", thr_otsu),
        ("mad", thr_mad),
    ]

    # 加入百分位数阈值。
    threshold_candidates.extend(
        (
            f"p{percentile}",
            float(np.percentile(env, percentile)),
        )
        for percentile in PERCENTILE_THRESHOLDS
    )

    # 逐个尝试候选阈值。
    for label, threshold in threshold_candidates:
        # 根据当前阈值检测动作段。
        segments = _segments_from_threshold(
            env,
            centers,
            threshold,
            n_samples,
        )

        # 保存当前候选结果。
        candidates.append(
            (label, threshold, segments)
        )

        # 检查当前结果是否合理。
        ok, _reason = validate(
            segments,
            n_samples,
        )

        # 找到合理结果后立即返回。
        if ok:
            return segments, threshold, label

    # 所有阈值都失败时，选择检测段数最多的结果。
    candidates.sort(
        key=lambda item: len(item[2]),
        reverse=True,
    )

    # 取得最接近预期的候选结果。
    label, threshold, segments = candidates[0]

    return segments, threshold, label


def _segments_from_threshold(
    env: np.ndarray,
    centers: np.ndarray,
    threshold: float,
    n_samples: int,
) -> list[tuple[int, int]]:
    """根据一个阈值提取动作段。"""

    # 高于阈值的位置认为是 active。
    mask = env > threshold

    # 将允许合并的秒数转换成 RMS 帧数。
    gap_frames = max(
        1,
        int(round(GAP_CLOSE_S * FS / HOP)),
    )

    # 合并动作内部的短间隔。
    if gap_frames > 0 and mask.any():
        mask = _close_short_gaps(
            mask,
            gap_frames,
        )

    # 保存原始采样点形式的动作区间。
    segments: list[tuple[int, int]] = []

    # 遍历所有连续 active 区间。
    for start_idx, end_idx in runs_of_true(mask):
        # 将包络开始位置转换成原始采样点。
        start_sample = (
            int(centers[start_idx] - WIN // 2)
            if start_idx < centers.size
            else 0
        )

        # 找到区间最后一个包络位置。
        last_idx = min(
            end_idx - 1,
            centers.size - 1,
        )

        # 将包络结束位置转换成原始采样点。
        end_sample = int(
            centers[last_idx] + WIN // 2
        )

        # 防止开始位置超出信号范围。
        start_sample = max(
            0,
            start_sample,
        )

        # 防止结束位置超出信号范围。
        end_sample = min(
            n_samples,
            end_sample,
        )

        # 将动作长度转换成秒。
        duration_s = (
            end_sample - start_sample
        ) / FS

        # 保留持续时间大致合理的动作段。
        if (
            MIN_ACTIVE_S
            <= duration_s
            <= MAX_ACTIVE_S + 1.0
        ):
            segments.append(
                (start_sample, end_sample)
            )

    return segments


def _close_short_gaps(
    mask: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """把 active 中间很短的 False gap 补成 True。"""

    # 复制 mask，避免修改原始数组。
    closed = mask.copy()

    # 从第一个位置开始检查。
    i = 0

    while i < closed.size:
        # 当前是 active 时直接前进。
        if closed[i]:
            i += 1
            continue

        # 找到连续 False 区间的结束位置。
        j = i

        while (
            j < closed.size
            and not closed[j]
        ):
            j += 1

        # 两侧都有 active 才算内部间隔。
        is_internal_gap = (
            i > 0
            and j < closed.size
        )

        # 将较短的内部间隔补成 active。
        if (
            is_internal_gap
            and (j - i) <= max_gap_frames
        ):
            closed[i:j] = True

        # 从当前间隔结束位置继续检查。
        i = j

    return closed


def validate(
    segments: list[tuple[int, int]],
    n_samples: int,
) -> tuple[bool, str]:
    """检查动作数量、时长和 rest gap 是否合理。"""

    # 动作段数量必须为4。
    if len(segments) != EXPECTED_ACTIVE:
        return (
            False,
            f"expected {EXPECTED_ACTIVE} "
            f"active segments, got {len(segments)}",
        )

    # 检查每个动作的持续时间。
    for start, end in segments:
        # 将采样点数量转换成秒。
        duration_s = (end - start) / FS

        # 动作时间超出范围时失败。
        if not (
            MIN_ACTIVE_S
            <= duration_s
            <= MAX_ACTIVE_S
        ):
            return (
                False,
                f"segment duration "
                f"{duration_s:.2f}s out of range",
            )

    # 从信号开始位置检查 rest。
    prev_end = 0

    # 检查每个动作前的 rest 时间。
    for start, end in segments:
        # 计算动作前的休息时间。
        rest_s = (start - prev_end) / FS

        # rest 太短时失败。
        if rest_s < MIN_REST_S:
            return (
                False,
                f"rest gap before sample {start} "
                f"shorter than {MIN_REST_S}s",
            )

        # 保存当前动作结束位置。
        prev_end = end

    # 计算最后一个动作后的休息时间。
    trailing_rest_s = (
        n_samples - prev_end
    ) / FS

    # 结尾休息时间太短时失败。
    if trailing_rest_s < MIN_REST_S:
        return (
            False,
            f"trailing rest shorter than "
            f"{MIN_REST_S}s",
        )

    return True, "ok"


def sampen(
    x: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
    max_n: int = 4000,
) -> float:
    """计算 Sample Entropy。"""

    # 转换成浮点数组。
    x = np.asarray(x, dtype=float)

    # 信号太长时进行等间隔下采样。
    if x.size > max_n:
        idx = np.linspace(
            0,
            x.size - 1,
            max_n,
        ).astype(int)

        x = x[idx]

    # 取得信号长度。
    n = x.size

    # 信号太短时无法计算。
    if n <= m + 1:
        return float("nan")

    # 计算标准差。
    sd = x.std()

    # 常数信号无法计算。
    if sd == 0:
        return float("nan")

    # 根据标准差设置匹配半径。
    radius = r_factor * sd

    def _phi(mm: int) -> int:
        """统计长度为 mm 的相似模板数量。"""

        # 创建滑动模板。
        templates = (
            np.lib.stride_tricks
            .sliding_window_view(x, mm)
        )

        # 初始化匹配数量。
        count = 0

        # 将每个模板与后续模板比较。
        for i in range(
            templates.shape[0] - 1
        ):
            # 使用 Chebyshev 距离。
            dist = np.max(
                np.abs(
                    templates[i + 1 :]
                    - templates[i]
                ),
                axis=1,
            )

            # 累加匹配数量。
            count += int(
                np.sum(dist < radius)
            )

        return count

    # 统计长度为 m 的模板匹配数。
    b_count = _phi(m)

    # 统计长度为 m+1 的模板匹配数。
    a_count = _phi(m + 1)

    # 没有匹配时无法计算。
    if b_count == 0 or a_count == 0:
        return float("nan")

    # 计算 Sample Entropy。
    return float(
        -np.log(a_count / b_count)
    )


def labels_from_segments(
    segments: list[tuple[int, int]],
    n_samples: int,
) -> np.ndarray:
    """生成逐采样点标签：rest=0，动作依次为1到4。"""

    # 默认所有采样点都是 rest。
    labels = np.zeros(
        n_samples,
        dtype=np.int8,
    )

    # 按动作出现顺序设置标签。
    for label, (start, end) in enumerate(
        sorted(segments),
        start=1,
    ):
        labels[start:end] = label

    return labels


def diag_plot(
    env: np.ndarray,
    centers: np.ndarray,
    threshold: float,
    segments: list[tuple[int, int]],
    sampen_per_seg: list[float],
    n_samples: int,
    title: str,
    output_path: Path,
) -> None:
    """保存 RMS envelope、阈值和动作段诊断图。"""

    # 将采样点转换成秒。
    t_env = centers / FS

    # 计算信号总时长。
    t_total = n_samples / FS

    # 创建画布。
    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    # 绘制 RMS 包络。
    ax.plot(
        t_env,
        env,
        color="steelblue",
        linewidth=0.8,
        label="RMS envelope",
    )

    # 绘制阈值线。
    ax.axhline(
        threshold,
        color="red",
        linewidth=0.8,
        linestyle="--",
        label=f"thr={threshold:.2e}",
    )

    # 取得包络最大值。
    ymax = (
        float(np.max(env))
        if env.size
        else 1.0
    )

    # 绘制每个动作段。
    for i, ((start, end), se) in enumerate(
        zip(
            sorted(segments),
            sampen_per_seg,
        ),
        start=1,
    ):
        # 使用阴影表示动作范围。
        ax.axvspan(
            start / FS,
            end / FS,
            color="orange",
            alpha=0.25,
        )

        # 显示动作编号和 Sample Entropy。
        ax.text(
            (start + end) / 2 / FS,
            ymax * 0.95,
            f"C{i}\nSE={se:.2f}",
            ha="center",
            va="top",
            fontsize=9,
        )

    # 设置图像范围和标签。
    ax.set_xlim(0, t_total)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("RMS")
    ax.set_title(title)

    # 显示图例。
    ax.legend(
        loc="upper right",
        fontsize=8,
    )

    # 调整布局。
    fig.tight_layout()

    # 创建图片输出目录。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 保存图片。
    fig.savefig(
        output_path,
        dpi=120,
    )

    # 关闭图片。
    plt.close(fig)


def process_file(
    src: Path,
) -> bool:
    """处理一个 filtered CSV 并输出标注结果。"""

    # 读取 filtered CSV。
    x = np.loadtxt(
        src,
        delimiter=",",
    )

    # 一维输入转换成单通道二维形式。
    if x.ndim == 1:
        x = x[:, None]

    # 取得总采样点数量。
    n_samples = x.shape[0]

    # 计算 RMS 包络。
    env, centers = rms_envelope(x)

    # 自动检测动作段。
    segments, threshold, threshold_label = (
        detect_segments(
            env,
            centers,
            n_samples,
        )
    )

    # 检查检测结果是否合理。
    ok, reason = validate(
        segments,
        n_samples,
    )

    # 找到方差最大的通道。
    most_active_ch = int(
        np.argmax(x.var(axis=0))
    )

    # 计算每个动作段的 Sample Entropy。
    sampen_per_seg = [
        sampen(
            x[
                start:end,
                most_active_ch,
            ]
        )
        for start, end in sorted(segments)
    ]

    # 去掉文件名中的 _filtered。
    base = src.stem.replace(
        "_filtered",
        "",
    )

    # 生成三个输出路径。
    out_labeled = src.with_name(
        f"{base}_labeled.csv"
    )

    out_segments = src.with_name(
        f"{base}_segments.csv"
    )

    # 诊断图保存在当前实验目录下的 pic 文件夹。
    out_plot = (
        src.parent
        / "pic"
        / f"{base}_diag.png"
    )

    # 检测成功时生成正常标签。
    if ok:
        labels = labels_from_segments(
            segments,
            n_samples,
        )

    # 检测失败时全部写成-1。
    else:
        labels = np.full(
            n_samples,
            -1,
            dtype=np.int8,
        )

    # 将标签添加到最后一列。
    labeled = np.hstack(
        [
            x,
            labels.reshape(-1, 1),
        ]
    )

    # 保存 labeled CSV。
    np.savetxt(
        out_labeled,
        labeled,
        delimiter=",",
        fmt=(
            ["%.8e"] * x.shape[1]
            + ["%d"]
        ),
    )

    # 保存 segments CSV。
    with out_segments.open(
        "w",
        encoding="utf-8",
    ) as f:
        # 写入表头。
        f.write(
            "start_sample,end_sample,"
            "label,duration_s,sampen\n"
        )

        # 成功时写入1到4。
        if ok:
            for label, (start, end) in enumerate(
                sorted(segments),
                start=1,
            ):
                se = sampen_per_seg[
                    label - 1
                ]

                duration_s = (
                    end - start
                ) / FS

                f.write(
                    f"{start},{end},{label},"
                    f"{duration_s:.4f},"
                    f"{se:.4f}\n"
                )

        # 失败时写入-1。
        else:
            for i, (start, end) in enumerate(
                sorted(segments)
            ):
                se = (
                    sampen_per_seg[i]
                    if i < len(sampen_per_seg)
                    else float("nan")
                )

                duration_s = (
                    end - start
                ) / FS

                f.write(
                    f"{start},{end},-1,"
                    f"{duration_s:.4f},"
                    f"{se:.4f}\n"
                )

    # 创建诊断图标题。
    status = (
        "OK"
        if ok
        else "FAIL: " + reason
    )

    title = (
        f"{src.name}  "
        f"thr={threshold_label}  "
        f"{status}"
    )

    # 保存诊断图。
    diag_plot(
        env=env,
        centers=centers,
        threshold=threshold,
        segments=segments,
        sampen_per_seg=sampen_per_seg,
        n_samples=n_samples,
        title=title,
        output_path=out_plot,
    )

    # 显示成功信息。
    if ok:
        print(
            f"[OK]   {src.name} "
            f"-> {out_labeled.name}"
        )

    # 显示失败和人工检查提示。
    else:
        print(
            f"[FAIL] {src.name} "
            f"({reason}) "
            f"-> {out_labeled.name} "
            f"(label=-1)",
            file=sys.stderr,
        )

        print(
            f"       MANUAL LABELING REQUIRED: "
            f"edit {out_segments.name} "
            f"or check {out_plot.name}",
            file=sys.stderr,
        )

    return ok


def process_directory(
    folder: Path,
) -> tuple[int, int]:
    """处理目录中的全部 filtered CSV。"""

    # 找到所有 filtered CSV。
    files = sorted(
        folder.glob("*_filtered.csv")
    )

    # 没有找到文件时给出提示。
    if not files:
        print(
            f"No *_filtered.csv in {folder}. "
            f"Run preprocessing first.",
            file=sys.stderr,
        )

        return 0, 0

    # 记录成功数量。
    n_ok = 0

    # 逐个调用 process_file()。
    for src in files:
        if process_file(src):
            n_ok += 1

    # 计算失败数量。
    n_fail = len(files) - n_ok

    return n_ok, n_fail


def main() -> None:
    """根据输入路径处理单个文件或整个目录。"""

    # 提供路径时使用命令行路径。
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

    # 没有提供路径时使用默认目录。
    else:
        input_path = DEFAULT_INPUT_PATH

    # 输入是单个文件时只处理一次。
    if input_path.is_file():
        ok = process_file(input_path)

        # 将 bool 转换成统计数量。
        n_ok = int(ok)
        n_fail = int(not ok)

    # 输入是目录时批量处理。
    elif input_path.is_dir():
        n_ok, n_fail = process_directory(
            input_path
        )

    # 路径不存在时停止。
    else:
        raise FileNotFoundError(
            f"Input path not found: {input_path}"
        )

    # 计算总文件数量。
    total = n_ok + n_fail

    # 输出最终汇总。
    print(
        f"\nSummary: success {n_ok} / "
        f"manual {n_fail} "
        f"(total {total})"
    )


if __name__ == "__main__":
    main()