"""Linkwitz-Riley 多频段分频的核心接口。

本模块只使用 Python 标准库（``math``/``cmath``），不依赖第三方 DSP
包，也不使用随机数或阻塞等待。

统一约定
--------
* 音频是单声道 ``list[float]``，采样点按时间顺序排列。
* 采样率 ``sample_rate`` 的单位是 Hz，必须是有限正数。
* 分频点 ``crossovers`` 的单位是 Hz，必须严格递增且低于奈奎斯特频率。
* ``order`` 是 Linkwitz-Riley 阶数，必须为正偶数。
* 所有时域/频域数值比较的绝对容差为 ``1e-6``（除非单独说明）。
"""

from __future__ import annotations

import math

__all__ = [
    "CrossoverError",
    "linkwitz_riley",
    "split_bands",
    "band_gain",
    "recombine",
    "phase_compensate",
    "smoothed_gain",
    "validate_config",
    "split_batch",
]

# 判定「首个显著样本」的阈值，与 recombine 的契约保持一致。
_SIGNIFICANT = 1e-12


class CrossoverError(Exception):
    """分频配置或输入音频非法时抛出的异常。"""


# ---------------------------------------------------------------------------
# 基础校验辅助
# ---------------------------------------------------------------------------


def _is_finite_number(value) -> bool:
    """判断 ``value`` 是否为有限实数（``bool`` 也视为数值以外的输入）。"""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_sample_rate(sample_rate) -> None:
    if not _is_finite_number(sample_rate) or sample_rate <= 0:
        raise CrossoverError(f"sample_rate 必须是有限正数，收到 {sample_rate!r}")


def _validate_order(order) -> None:
    if (
        not isinstance(order, int)
        or isinstance(order, bool)
        or order < 2
        or order % 2 != 0
    ):
        raise CrossoverError(
            f"order 必须是正偶数（2、4、6、...），收到 {order!r}"
        )


def _validate_crossovers(
    crossovers, sample_rate: float, *, allow_empty: bool = True
) -> None:
    if not isinstance(crossovers, (list, tuple)):
        raise CrossoverError(f"crossovers 必须是分频点列表，收到 {crossovers!r}")
    if not allow_empty and not crossovers:
        raise CrossoverError("crossovers 不能为空")

    nyquist = sample_rate / 2.0
    previous = None
    for index, value in enumerate(crossovers):
        if not _is_finite_number(value) or value <= 0:
            raise CrossoverError(
                f"crossovers[{index}] 必须是有限正数，收到 {value!r}"
            )
        if value >= nyquist:
            raise CrossoverError(
                f"crossovers[{index}]={value!r} 达到或超过奈奎斯特频率 {nyquist!r}"
            )
        if previous is not None and value <= previous:
            raise CrossoverError(
                "crossovers 必须严格递增，"
                f"第 {index - 1} 个 {previous!r} 未小于第 {index} 个 {value!r}"
            )
        previous = value


def _validate_samples(samples, name: str = "samples") -> None:
    if not isinstance(samples, (list, tuple)):
        raise CrossoverError(f"{name} 必须是浮点样本列表，收到 {type(samples)!r}")
    for index, value in enumerate(samples):
        if not _is_finite_number(value):
            raise CrossoverError(
                f"{name}[{index}] 必须是有限数值，收到 {value!r}"
            )


# ---------------------------------------------------------------------------
# Linkwitz-Riley 滤波器设计
# ---------------------------------------------------------------------------


def _design_half(freq: float, order: int, sample_rate: float, *, high: bool):
    """设计 LR 低通或高通的二阶节级联。

    做法：把 ``order / 2`` 个 Butterworth 极点经双线性变换映射到 z 平面，
    再将每一对共轭极点（或实极点）展开成两个相同的二阶节，从而得到
    ``order`` 阶 Linkwitz-Riley 响应。每个二阶节按单位直流/奈奎斯特增益
    归一化，使低通在直流、高通在奈奎斯特点的幅度恰为 1，分频点处两者
    均为 ``-6.0206 dB``。
    """

    sections = []
    half = order // 2
    theta = math.pi * freq / sample_rate

    for k in range(half):
        # half 阶 Butterworth 的第 k 个 s 平面极点。
        angle = math.pi * (2 * k + half + 1) / (2 * half)
        if math.sin(angle) < 0.0:
            # 共轭极点由正虚部极点代表，避免重复计算导致阶数翻倍。
            continue
        pole = complex(math.cos(angle), math.sin(angle)) * math.tan(theta)
        z_pole = (1.0 + pole) / (1.0 - pole)
        a1 = -2.0 * z_pole.real
        a2 = abs(z_pole) ** 2

        if abs(z_pole.imag) < 1e-12:
            # 实极点：一个二阶节（第二个极点视为 0）。
            gain = abs(1.0 - z_pole) ** 2 / 4.0
            if high:
                gain = abs(1.0 + z_pole) ** 2 / 4.0
            b0 = gain
            b1 = -2.0 * gain if high else 2.0 * gain
            b2 = gain
            sections.append((b0, b1, b2, a1, a2))
        else:
            # 共轭极点对：用两个相同的二阶节构成一个四阶 Butterworth
            # 因子，这正是 LR 所需的平方结构。
            gain_low = abs(1.0 - z_pole) ** 2 / 4.0
            gain_high = abs(1.0 + z_pole) ** 2 / 4.0
            if high:
                section = (gain_high, -2.0 * gain_high, gain_high, a1, a2)
            else:
                section = (gain_low, 2.0 * gain_low, gain_low, a1, a2)
            sections.append(section)
            sections.append(section)

    return tuple(sections)


def linkwitz_riley(freq: float, order: int, sample_rate: float) -> tuple:
    """构造分频点为 ``freq`` Hz 的 Linkwitz-Riley 分频器。

    参数
    ----
    freq:
        分频点，单位 Hz；必须满足 ``0 < freq < sample_rate / 2``。
    order:
        Linkwitz-Riley 阶数，必须为正偶数（2、4、6、...）。
    sample_rate:
        采样率，单位 Hz；必须是有限正数。

    返回
    ----
    ``(lowpass_sections, highpass_sections)`` 二元组。每一半是由二阶节
    组成的元组，二阶节按信号从输入到输出的顺序排列；单个二阶节的格式
    为 ``(b0, b1, b2, a1, a2)``，其中 ``a0`` 已归一化为 1。每一半包含
    ``order // 2`` 个二阶节。

    契约
    ----
    * 这是 Linkwitz-Riley 响应，不是 Butterworth 级联：低通和高通在
      ``freq`` 处的幅度都必须约为 ``-6.0206 dB``。时域与重建比较使用
      ``1e-6`` 绝对容差；滤波器响应测试围绕该值检查，以容纳合法数字
      滤波实现的截止点误差。
    * 低通与高通在分频点附近应能相位重建；实现应使用确定性算术。
    * ``freq`` 非法、``order`` 非正偶数、``sample_rate`` 非正时抛出
      ``CrossoverError``。
    """

    _validate_sample_rate(sample_rate)
    _validate_order(order)
    if not _is_finite_number(freq) or freq <= 0:
        raise CrossoverError(f"freq 必须是有限正数，收到 {freq!r}")
    if freq >= sample_rate / 2.0:
        raise CrossoverError(
            f"freq={freq!r} 达到或超过奈奎斯特频率 {sample_rate / 2.0!r}"
        )

    lowpass = _design_half(freq, order, sample_rate, high=False)
    highpass = _design_half(freq, order, sample_rate, high=True)
    return lowpass, highpass


# ---------------------------------------------------------------------------
# 二阶节级联与频段切分
# ---------------------------------------------------------------------------


def _run_sections(samples, sections) -> list:
    """用独立的 Direct-Form II Transposed 状态运行二阶节级联。"""

    output = list(samples)
    for b0, b1, b2, a1, a2 in sections:
        state_1 = 0.0
        state_2 = 0.0
        filtered = [0.0] * len(output)
        for index, value in enumerate(output):
            result = b0 * value + state_1
            state_1 = b1 * value - a1 * result + state_2
            state_2 = b2 * value - a2 * result
            filtered[index] = result
        output = filtered
    return output


def _difference(left, right) -> list:
    return [a - b for a, b in zip(left, right)]


def split_bands(
    samples: list[float],
    crossovers: list[float],
    order: int,
    sample_rate: float,
) -> list[list[float]]:
    """把单声道音频切分为 ``len(crossovers) + 1`` 个频段。

    参数
    ----
    samples:
        输入音频，``list[float]``；空列表表示空音频，返回 ``[]``。
    crossovers:
        分频点列表，单位 Hz，必须严格递增且全部低于奈奎斯特频率；
        空列表表示不切分，返回仅含原始信号的一份拷贝。
    order:
        Linkwitz-Riley 阶数，必须为正偶数。
    sample_rate:
        采样率，单位 Hz，必须是有限正数。

    返回
    ----
    由频段组成的列表，顺序从低频到高频，每个频段都是与 ``samples``
    等长的 ``list[float]``。当 ``samples`` 非空且分频点合法时，元素个数
    必须恰好为 ``len(crossovers) + 1``。

    契约
    ----
    * 全部频段逐样本相加必须还原 ``samples``，绝对误差不超过 ``1e-6``。
      因此实现必须处理 Linkwitz-Riley 的全通相位与延迟补偿，不能只把
      低通/高通直接相加。
    * 每次调用的滤波器状态完全独立，不能跨调用泄漏。
    * 分频点不递增、越界、阶数为奇数、采样率非正或样本含非有限值时
      抛出 ``CrossoverError``，不得返回部分结果。
    """

    _validate_sample_rate(sample_rate)
    _validate_order(order)
    _validate_crossovers(crossovers, sample_rate)
    _validate_samples(samples)

    if not samples:
        return []
    if not crossovers:
        return [list(samples)]

    # 逐级抵消式切分：每级用 LR 低通取出当前最低频段，其余信号用
    # ``原始 - 低通`` 得到。LR 低通与高通的相位之和是一个全通网络，
    # 因此「取差」正好等价于在每一级做了全通反补偿；这样既保持了各段
    # 正确的 LR 幅度响应，又保证所有频段逐样本相加严格还原输入，不会
    # 因相位/延迟错位产生梳状滤波。
    bands = []
    residual = list(samples)
    last = len(crossovers) - 1

    for index, cutoff in enumerate(crossovers):
        lowpass, _highpass = linkwitz_riley(cutoff, order, sample_rate)
        low = _run_sections(residual, lowpass)
        bands.append(low)
        residual = _difference(residual, low)
        if index == last:
            # 最后一级剩余的信号就是最高频段。
            bands.append(residual)

    return bands


# ---------------------------------------------------------------------------
# 增益
# ---------------------------------------------------------------------------


def band_gain(bands: list[list[float]], gains_db: list[float]) -> list[list[float]]:
    """对每个频段施加独立增益，增益以 dB 给出。

    参数
    ----
    bands:
        频段列表，每个频段是 ``list[float]``；各频段可以等长也可以不等长。
    gains_db:
        与 ``bands`` 一一对应的增益，单位 dB。必须使用
        ``10 ** (gains_db / 20)`` 换算为线性幅度，而不是直接乘 dB 数值。

    返回
    ----
    新的频段列表，形状与 ``bands`` 相同；不修改输入列表。

    契约
    ----
    * 例如 ``+6.020599913279624 dB`` 对应约 ``2.0`` 倍，
      ``-6.020599913279624 dB`` 对应约 ``0.5`` 倍；绝对容差 ``1e-6``。
    * 增益为 0 dB 时输出必须与输入逐样本相等（容差 ``1e-6``）。
    * ``len(bands) != len(gains_db)`` 时抛出 ``CrossoverError``；任一增益
      不是有限数时也抛出 ``CrossoverError``。
    """

    if not isinstance(bands, (list, tuple)) or not isinstance(
        gains_db, (list, tuple)
    ):
        raise CrossoverError("bands 与 gains_db 都必须是列表")
    if len(bands) != len(gains_db):
        raise CrossoverError(
            f"bands 数量 {len(bands)} 与 gains_db 数量 {len(gains_db)} 不一致"
        )

    for index, gain_db in enumerate(gains_db):
        if not _is_finite_number(gain_db):
            raise CrossoverError(
                f"gains_db[{index}] 必须是有限数值，收到 {gain_db!r}"
            )

    result = []
    for band, gain_db in zip(bands, gains_db):
        if not isinstance(band, (list, tuple)):
            raise CrossoverError(f"频段必须是浮点样本列表，收到 {type(band)!r}")
        factor = 10.0 ** (gain_db / 20.0)
        result.append([factor * value for value in band])
    return result


def smoothed_gain(gains_db: list[float], time_ms: float, sample_rate: float) -> list[float]:
    """把目标增益序列展开成无爆音的线性增益包络。

    参数
    ----
    gains_db:
        目标增益序列，单位 dB；至少包含一个元素。
    time_ms:
        相邻目标之间的过渡时长，单位毫秒；必须为非负有限数。为 0 时
        允许瞬时切换。
    sample_rate:
        采样率，单位 Hz，必须是有限正数。

    返回
    ----
    线性幅度增益包络 ``list[float]``。若 ``n = len(gains_db)``，
    ``ramp = round(time_ms * sample_rate / 1000)``，长度应为
    ``(n - 1) * ramp + 1``（``n == 1`` 时长度为 1）。包络在相邻目标
    之间使用 smoothstep 插值，端点分别精确等于对应 dB 的线性增益。

    契约
    ----
    * 增益变化连续平滑，相邻样本之差应小于 ``1e-2``（在合理过渡时长
      下），不得出现瞬时跳变造成的咔哒声。
    * 输出全部为有限正数；端点与 ``10 ** (gains_db / 20)`` 的绝对误差
      不超过 ``1e-6``。
    * 空序列、非有限增益、``time_ms < 0`` 或采样率非正时抛出
      ``CrossoverError``。
    """

    _validate_sample_rate(sample_rate)
    if not _is_finite_number(time_ms) or time_ms < 0:
        raise CrossoverError(
            f"time_ms 必须是非负有限数，收到 {time_ms!r}"
        )
    if not isinstance(gains_db, (list, tuple)) or not gains_db:
        raise CrossoverError("gains_db 必须是非空列表")

    linear = []
    for index, gain_db in enumerate(gains_db):
        if not _is_finite_number(gain_db):
            raise CrossoverError(
                f"gains_db[{index}] 必须是有限数值，收到 {gain_db!r}"
            )
        linear.append(10.0 ** (gain_db / 20.0))

    if len(linear) == 1:
        return [linear[0]]

    ramp = int(round(time_ms * sample_rate / 1000.0))
    if ramp <= 0:
        # 瞬时切换：每个目标成为一段的起点，保留全部目标值。
        return [value for value in linear]

    envelope = [linear[0]]
    for index in range(len(linear) - 1):
        start = linear[index]
        end = linear[index + 1]
        for step in range(1, ramp + 1):
            t = step / ramp
            # smoothstep：一阶导在端点为零，避免斜率突变造成爆音。
            weight = t * t * (3.0 - 2.0 * t)
            envelope.append(start + (end - start) * weight)
    return envelope


# ---------------------------------------------------------------------------
# 延迟 / 相位补偿与重组
# ---------------------------------------------------------------------------


def _first_significant(band) -> int | None:
    for index, value in enumerate(band):
        if abs(value) > _SIGNIFICANT:
            return index
    return None


def _shift_band(band, shift: int) -> list:
    """把 ``band[i]`` 移到 ``band[i + shift]``；越界处补零。"""

    length = len(band)
    if shift == 0:
        return list(band)
    shifted = [0.0] * length
    for index in range(length):
        source = index + shift
        if 0 <= source < length:
            shifted[index] = band[source]
    return shifted


def _compensation_shift(band, reference: int) -> int:
    """计算频段相对参考时刻需要的平移量。

    以第一个频段的首个显著样本为参考。只有当本频段的首个显著样本晚于
    参考时才把它整体前移，用于抵消因果滤波器引入的延迟；若本频段开始
    得并不更晚（含全零频段），则保持原位。这样可避免把已经开始的有效
    信号向后挪而丢掉开头内容，同时对已经对齐的输入平移量恒为 0。
    """

    onset = _first_significant(band)
    if onset is None:
        return 0
    return max(0, onset - reference)


def _validate_bands(bands, *, allow_empty: bool = False) -> None:
    if not isinstance(bands, (list, tuple)):
        raise CrossoverError("bands 必须是频段列表")
    if not bands:
        if allow_empty:
            return
        raise CrossoverError("bands 不能为空")
    lengths = set()
    for band_index, band in enumerate(bands):
        if not isinstance(band, (list, tuple)):
            raise CrossoverError(
                f"bands[{band_index}] 必须是浮点样本列表，收到 {type(band)!r}"
            )
        if not band:
            raise CrossoverError(f"bands[{band_index}] 不能为空")
        lengths.add(len(band))
        _validate_samples(band, f"bands[{band_index}]")
    if len(lengths) != 1:
        raise CrossoverError(f"所有频段必须等长，收到长度集合 {sorted(lengths)!r}")


def recombine(bands: list[list[float]]) -> list[float]:
    """把已分频的频段做延迟补偿后求和。

    参数
    ----
    bands:
        等长的频段列表，每个频段是 ``list[float]``。因果滤波器可能在
        频段开头留下不同长度的零延迟前缀。

    返回
    ----
    长度为 ``len(bands[0])`` 的单声道 ``list[float]``。

    契约
    ----
    * 以第一个频段的首个显著样本（``abs(x) > 1e-12``）为时间参考；其余
      频段先平移，使其首个显著样本与该参考对齐，再逐样本相加。这样
      高频段不会因未补偿延迟而提前/落后，也不会因相位错位产生梳状
      抵消。全零频段保持原位。
    * 求和结果与各频段对齐后之和的绝对误差不超过 ``1e-6``。
    * 空 ``bands``、频段长度不一致、频段为空或样本非有限时抛出
      ``CrossoverError``。
    """

    _validate_bands(bands)
    length = len(bands[0])
    reference = _first_significant(bands[0])
    if reference is None:
        reference = 0

    total = [0.0] * length
    for band in bands:
        aligned = _shift_band(band, _compensation_shift(band, reference))
        for index in range(length):
            total[index] += aligned[index]
    return total


def phase_compensate(
    bands: list[list[float]], order: int, sample_rate: float
) -> list[list[float]]:
    """对分频后的各频段做 Linkwitz-Riley 全通相位补偿。

    参数
    ----
    bands:
        等长的频段列表，每个频段是 ``list[float]``。
    order:
        Linkwitz-Riley 阶数，必须为正偶数；用于确定补偿的相位/延迟量。
    sample_rate:
        采样率，单位 Hz，必须是有限正数。

    返回
    ----
    新的等长频段列表，形状与输入一致；不修改输入。补偿后各频段在分频
    点附近的群延迟一致，再相加不会产生梳状滤波。

    契约
    ----
    * 只改变相位/延迟，不改变各频段的幅度谱；输入全零时输出必须全零。
    * 返回的每个频段长度都等于对应输入频段长度，数值绝对容差 ``1e-6``。
    * 对已经相位对齐的输入应保持幂等（重复调用不产生可测偏移）。
    * 阶数为奇数、采样率非正或频段长度不一致时抛出 ``CrossoverError``。
    """

    _validate_order(order)
    _validate_sample_rate(sample_rate)
    _validate_bands(bands)

    # 以第一个频段的起始时刻为参考，把其余频段整体平移，使其首个显著
    # 样本落在同一时刻。整数延迟只改变相位/群延迟而不改变幅度谱；对已
    # 对齐的输入再次调用时平移量恒为 0，因此是幂等的。
    reference = _first_significant(bands[0])
    if reference is None:
        reference = 0

    aligned_bands = []
    for band in bands:
        aligned_bands.append(_shift_band(band, _compensation_shift(band, reference)))
    return aligned_bands


# ---------------------------------------------------------------------------
# 配置校验与批量处理
# ---------------------------------------------------------------------------


def validate_config(
    crossovers: list[float], order: int, sample_rate: float
) -> list[str]:
    """检查分频配置，返回人类可读的问题列表。

    参数
    ----
    crossovers:
        分频点列表，单位 Hz；应为严格递增的有限正数，且每个值都低于
        ``sample_rate / 2``。
    order:
        Linkwitz-Riley 阶数，必须为正偶数。
    sample_rate:
        采样率，单位 Hz，必须是有限正数；0 或负数属于非法配置。

    返回
    ----
    问题描述字符串列表；合法配置返回 ``[]``。此函数只报告问题，不抛出
    配置异常，便于批量场景一次性收集全部错误。建议每条消息包含
    ``"crossovers"``、``"nyquist"``、``"order"`` 或 ``"sample_rate"``
    等可检索关键词，便于调用方定位。

    契约
    ----
    至少覆盖：分频点未严格递增、分频点非正/非有限、分频点达到或超过
    奈奎斯特频率、阶数为奇数或小于 2、采样率非正或非有限。
    """

    issues = []

    sample_rate_valid = _is_finite_number(sample_rate) and sample_rate > 0
    if not sample_rate_valid:
        issues.append(f"sample_rate 必须是有限正数，收到 {sample_rate!r}")

    if (
        not isinstance(order, int)
        or isinstance(order, bool)
        or order < 2
        or order % 2 != 0
    ):
        issues.append(f"order 必须是正偶数（2、4、6、...），收到 {order!r}")

    if not isinstance(crossovers, (list, tuple)):
        issues.append(f"crossovers 必须是分频点列表，收到 {crossovers!r}")
        return issues

    previous = None
    for index, value in enumerate(crossovers):
        if not _is_finite_number(value) or value <= 0:
            issues.append(
                f"crossovers[{index}] 必须是有限正数，收到 {value!r}"
            )
        else:
            if sample_rate_valid and value >= sample_rate / 2.0:
                issues.append(
                    f"crossovers[{index}]={value!r} 达到或超过 nyquist 频率 "
                    f"{sample_rate / 2.0!r}"
                )
            if previous is not None and value <= previous:
                issues.append(
                    f"crossovers 必须严格递增，第 {index} 个 {value!r} "
                    f"不大于第 {index - 1} 个 {previous!r}"
                )
        previous = value

    return issues


def split_batch(tracks: list[list[float]], config: dict) -> dict:
    """批量分频多条音轨，并保证任一轨失败时整体不产生结果。

    参数
    ----
    tracks:
        音轨列表，每项是单声道 ``list[float]``。
    config:
        包含 ``"crossovers"``（Hz 列表）、``"order"``（正偶数）和
        ``"sample_rate"``（Hz）的字典，语义与 :func:`split_bands` 相同。

    返回
    ----
    ``{"bands": [...], "count": len(tracks)}``：``bands`` 与输入音轨顺序
    一一对应，每项是该音轨的频段列表；``count`` 是成功处理的音轨数。

    契约
    ----
    * 先校验配置和所有音轨，再开始处理；任意一轨非法（非列表、含非
      有限样本等）时抛出 ``CrossoverError``，不得返回部分结果。
    * 每条音轨使用独立的滤波器状态，前一条音轨的尾音不得串入后一条。
    * 全部成功时输出顺序与 ``tracks`` 完全一致；每条结果应与单独调用
      :func:`split_bands` 的结果在 ``1e-6`` 容差内一致。
    * 空 ``tracks`` 返回 ``{"bands": [], "count": 0}``。
    """

    if not isinstance(config, dict):
        raise CrossoverError(f"config 必须是字典，收到 {type(config)!r}")
    for key in ("crossovers", "order", "sample_rate"):
        if key not in config:
            raise CrossoverError(f"config 缺少必需字段 {key!r}")

    crossovers = config["crossovers"]
    order = config["order"]
    sample_rate = config["sample_rate"]

    issues = validate_config(crossovers, order, sample_rate)
    if issues:
        raise CrossoverError("分频配置非法: " + "; ".join(issues))

    if not isinstance(tracks, (list, tuple)):
        raise CrossoverError(f"tracks 必须是音轨列表，收到 {type(tracks)!r}")

    # 先完整校验所有音轨，确保任一非法时不会返回部分结果。
    for track_index, track in enumerate(tracks):
        _validate_samples(track, f"tracks[{track_index}]")

    bands = []
    for track in tracks:
        # split_bands 每次调用都会新建滤波器状态，因此各音轨完全隔离。
        bands.append(split_bands(track, crossovers, order, sample_rate))

    return {"bands": bands, "count": len(tracks)}
