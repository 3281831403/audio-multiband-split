"""audio-multiband-split 的 15 个验收测试（初始全部为红）。"""

from __future__ import annotations

import cmath
import math

import pytest

from multiband.crossover import (
    CrossoverError,
    band_gain,
    linkwitz_riley,
    phase_compensate,
    recombine,
    smoothed_gain,
    split_bands,
    split_batch,
    validate_config,
)

TOL = 1e-6
SAMPLE_RATE = 48_000.0
CONFIG = {
    "crossovers": [1_200.0],
    "order": 4,
    "sample_rate": SAMPLE_RATE,
}


def _assert_close(actual, expected, tol=TOL):
    assert len(actual) == len(expected), (
        f"长度不一致: {len(actual)} != {len(expected)}"
    )
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert abs(got - want) <= tol, (
            f"样本 {index} 不匹配: {got!r} != {want!r} (tol={tol})"
        )


def _assert_bands_close(actual, expected, tol=TOL):
    assert len(actual) == len(expected), (
        f"频段数不一致: {len(actual)} != {len(expected)}"
    )
    for band_index, (got_band, want_band) in enumerate(zip(actual, expected)):
        assert len(got_band) == len(want_band), (
            f"频段 {band_index} 长度不一致: {len(got_band)} != {len(want_band)}"
        )
        for sample_index, (got, want) in enumerate(zip(got_band, want_band)):
            assert abs(got - want) <= tol, (
                f"频段 {band_index} 样本 {sample_index} 不匹配: "
                f"{got!r} != {want!r} (tol={tol})"
            )


def _tone(partials, n=256, sample_rate=SAMPLE_RATE, dc=0.2):
    """确定性的多正弦测试信号，不使用随机数。"""

    return [
        dc
        + sum(
            amplitude * math.sin(2.0 * math.pi * freq * index / sample_rate)
            for freq, amplitude in partials
        )
        for index in range(n)
    ]


def _cascade_gain(sections, freq, sample_rate):
    """计算若干二阶节级联在 freq 处的幅度响应。"""

    z_inv = cmath.exp(-2j * math.pi * freq / sample_rate)
    response = 1.0 + 0.0j
    for b0, b1, b2, a1, a2 in sections:
        numerator = b0 + b1 * z_inv + b2 * z_inv * z_inv
        denominator = 1.0 + a1 * z_inv + a2 * z_inv * z_inv
        response *= numerator / denominator
    return abs(response)


def test_linkwitz_riley_has_minus_6db_at_crossover():
    """缺陷 1：Butterworth 级联在分频点约为 -3 dB，LR 应为 -6 dB。"""

    lowpass, highpass = linkwitz_riley(1_000.0, 4, SAMPLE_RATE)
    low_db = 20.0 * math.log10(_cascade_gain(lowpass, 1_000.0, SAMPLE_RATE))
    high_db = 20.0 * math.log10(_cascade_gain(highpass, 1_000.0, SAMPLE_RATE))

    assert -6.8 < low_db < -5.4, f"低通在分频点应为 -6 dB，实际 {low_db:.3f} dB"
    assert -6.8 < high_db < -5.4, f"高通在分频点应为 -6 dB，实际 {high_db:.3f} dB"


def test_split_bands_rejects_non_increasing_crossovers():
    """缺陷 2：分频点不递增时必须报错，不能产生重叠频段。"""

    issues = validate_config([2_000.0, 1_000.0], 4, SAMPLE_RATE)
    assert issues, "非递增分频点必须出现在 validate_config 的问题列表里"

    with pytest.raises(CrossoverError):
        split_bands([0.0] * 32, [2_000.0, 1_000.0], 4, SAMPLE_RATE)


def test_split_then_recombine_has_no_comb_filtering():
    """缺陷 3：缺少全通/相位补偿时，重组会出现梳状滤波。"""

    samples = _tone([(440.0, 0.35), (3_100.0, 0.20)], n=512)
    bands = split_bands(samples, [1_200.0], 4, SAMPLE_RATE)

    assert len(bands) == 2
    assert all(len(band) == len(samples) for band in bands)
    for band in bands:
        assert all(math.isfinite(value) for value in band)

    total = [sum(band[index] for band in bands) for index in range(len(samples))]
    _assert_close(total, samples)

    rebuilt = recombine(bands)
    _assert_close(rebuilt, samples)

    aligned = phase_compensate(bands, 4, SAMPLE_RATE)
    assert len(aligned) == 2
    assert all(len(band) == len(samples) for band in aligned)


def test_odd_order_rejected_by_linkwitz_riley():
    """缺陷 4：奇数阶无法构成完整 LR 二阶节，必须拒绝。"""

    with pytest.raises(CrossoverError):
        linkwitz_riley(1_000.0, 3, SAMPLE_RATE)


def test_band_gain_converts_db_to_linear():
    """缺陷 5：dB 必须先换算成线性倍数，不能直接乘 dB 数值。"""

    bands = [[1.0, -1.0], [0.5, 0.5]]
    result = band_gain(bands, [6.020599913279624, -6.020599913279624])

    assert result[0][0] == pytest.approx(2.0, abs=TOL)
    assert result[0][1] == pytest.approx(-2.0, abs=TOL)
    assert result[1][0] == pytest.approx(0.25, abs=TOL)
    assert result[1][1] == pytest.approx(0.25, abs=TOL)


def test_smoothed_gain_ramps_without_clicks():
    """缺陷 6：增益直接跳变会产生爆音，包络必须连续平滑。"""

    ramp = smoothed_gain([-24.0, 0.0, -12.0], 5.0, SAMPLE_RATE)

    assert len(ramp) == 481
    assert ramp[0] == pytest.approx(10.0 ** (-24.0 / 20.0), abs=TOL)
    assert ramp[-1] == pytest.approx(10.0 ** (-12.0 / 20.0), abs=TOL)
    assert all(math.isfinite(value) and value > 0.0 for value in ramp)

    largest_step = max(abs(ramp[index + 1] - ramp[index]) for index in range(len(ramp) - 1))
    assert largest_step < 0.01, f"增益包络存在跳变: {largest_step:.6f}"


def test_split_bands_rejects_crossover_above_nyquist():
    """缺陷 7：分频点达到/超过奈奎斯特频率必须报错。"""

    issues = validate_config([5_000.0], 4, 8_000.0)
    assert issues, "超奈奎斯特分频点必须出现在 validate_config 的问题列表里"

    with pytest.raises(CrossoverError):
        split_bands([0.0] * 32, [5_000.0], 4, 8_000.0)


def test_recombine_compensates_band_delay():
    """缺陷 8：重组合必须先补偿各段延迟，否则高频提前、瞬态错位。"""

    low = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    high = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0]

    result = recombine([low, high])

    assert len(result) == 6
    assert result[2] == pytest.approx(1.5, abs=TOL)
    assert result[3] == pytest.approx(0.0, abs=TOL)


def test_split_batch_isolates_filter_state_between_tracks():
    """缺陷 9：批量处理不得共用滤波器状态，否则前轨尾音会串入后轨。"""

    active = [0.9, -0.4, 0.7, -0.2] + [0.0] * 28
    silence = [0.0] * 32

    result = split_batch([active, silence], CONFIG)

    assert result["count"] == 2
    assert len(result["bands"]) == 2
    assert any(abs(value) > 0.0 for band in result["bands"][0] for value in band)
    for band in result["bands"][1]:
        assert max((abs(value) for value in band), default=0.0) <= 1e-9

    with pytest.raises(CrossoverError):
        split_batch([active, [0.0, float("nan")]], CONFIG)


def test_zero_sample_rate_is_rejected():
    """缺陷 10：sample_rate=0 必须报配置错误，不能触发除零。"""

    issues = validate_config([1_000.0], 4, 0.0)
    assert issues, "sample_rate=0 必须出现在 validate_config 的问题列表里"

    with pytest.raises(CrossoverError):
        split_bands([0.1], [1_000.0], 4, 0.0)


def test_single_crossover_returns_two_bands():
    """组合：一个分频点应得到低、高两个频段。"""

    samples = _tone([(300.0, 0.3)], n=128)
    bands = split_bands(samples, [1_200.0], 4, SAMPLE_RATE)

    assert len(bands) == 2
    assert all(len(band) == len(samples) for band in bands)


def test_zero_db_gains_are_identity():
    """组合：所有增益为 0 dB 时，band_gain 输出应等于输入。"""

    bands = [
        [0.1, -0.2, 0.3],
        [0.0, 0.5, -0.5],
        [-1.0, 0.25, 0.0],
    ]
    result = band_gain(bands, [0.0, 0.0, 0.0])

    _assert_bands_close(result, bands)


def test_split_bands_rejects_odd_order():
    """组合：split_bands 遇到奇数阶必须抛出 CrossoverError。"""

    with pytest.raises(CrossoverError):
        split_bands([0.1] * 16, [1_000.0], 3, SAMPLE_RATE)


def test_empty_audio_returns_empty():
    """组合：空音频返回空结果，而不是 N+1 个空频段。"""

    assert split_bands([], [1_000.0], 4, SAMPLE_RATE) == []


def test_split_batch_preserves_successful_track_order():
    """组合：批量全部成功时，结果顺序必须与输入音轨顺序一致。"""

    tracks = [
        _tone([(220.0, 0.3)], n=128),
        _tone([(880.0, 0.2)], n=96),
        [0.0] * 40,
    ]
    result = split_batch(tracks, CONFIG)

    assert result["count"] == len(tracks)
    assert len(result["bands"]) == len(tracks)
    assert [len(per_track[0]) for per_track in result["bands"]] == [128, 96, 40]

    for index, track in enumerate(tracks):
        expected = split_bands(
            track,
            CONFIG["crossovers"],
            CONFIG["order"],
            CONFIG["sample_rate"],
        )
        _assert_bands_close(result["bands"][index], expected)
