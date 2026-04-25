# test_sionna_tools.py
import os
import numpy as np
from sionna_tools import simulate_channel, compute_time_correlation, plot_time_series, plot_correlation_curve
from utils.path_tool import get_abs_path
from utils.config_handler import agent_conf

def test_basic_generation():
    print("\n[TEST 1] 基本信道生成")

    result = simulate_channel(
        model="A",
        speed=3.0,
        batch_size=4,
        traj_len=10,
        fft_size=64,
    )

    H = result["H"]

    print(f"  H shape: {H.shape}")

    assert isinstance(H, np.ndarray), "H 不是 numpy 数组"
    assert np.iscomplexobj(H), "H 不是复数"
    assert H.ndim == 3, "维度应为 [B, T, FFT]"

    print("  ✔ 基本生成通过")


def test_time_correlation_low_speed():
    print("\n[TEST 2] 低速时间相关性（应接近1）")

    result = simulate_channel(
        model="A",
        speed=3.0,
        batch_size=6,
        traj_len=14,
    )

    stats = compute_time_correlation(result["H"])

    print(f"  平均相关性: {stats['mean_corr']:.4f}")

    assert stats["mean_corr"] > 0.9, "低速相关性不够高（异常）"

    print("  ✔ 低速相关性正常")


def test_time_correlation_high_speed():
    print("\n[TEST 3] 高速时间相关性（应下降）")

    result_low = simulate_channel(model="A", speed=3.0)
    result_high = simulate_channel(model="A", speed=30.0)

    corr_low = compute_time_correlation(result_low["H"])["mean_corr"]
    corr_high = compute_time_correlation(result_high["H"])["mean_corr"]

    print(f"  低速 corr: {corr_low:.4f}")
    print(f"  高速 corr: {corr_high:.4f}")

    assert corr_high < corr_low, "高速相关性没有下降（不符合物理规律）"

    print("  ✔ 多普勒效应正确")


def test_model_difference():
    print("\n[TEST 4] 不同CDL模型差异")

    result_A = simulate_channel(model="A", speed=30.0)
    result_D = simulate_channel(model="D", speed=30.0)

    corr_A = compute_time_correlation(result_A["H"])["mean_corr"]
    corr_D = compute_time_correlation(result_D["H"])["mean_corr"]

    print(f"  CDL-A corr: {corr_A:.4f}")
    print(f"  CDL-D corr: {corr_D:.4f}")

    assert abs(corr_A - corr_D) > 1e-3, "不同模型没有差异（异常）"

    print("  ✔ 模型差异存在")


def test_stability():
    print("\n[TEST 5] 多次运行稳定性")

    corrs = []

    for i in range(3):
        result = simulate_channel(model="A", speed=5.0)
        corr = compute_time_correlation(result["H"])["mean_corr"]
        corrs.append(corr)

    print(f"  多次结果: {[round(c,4) for c in corrs]}")

    std = np.std(corrs)
    print(f"  标准差: {std:.6f}")

    assert std < 0.05, "结果波动过大（不稳定）"

    print("  ✔ 稳定性正常")


def run_all_tests():
    print("=" * 60)
    print(" Sionna Tools 测试开始")
    print("=" * 60)

    test_basic_generation()
    test_time_correlation_low_speed()
    test_time_correlation_high_speed()
    test_model_difference()
    test_stability()

    print("\n" + "=" * 60)
    print(" ✅ 所有测试通过")
    print("=" * 60)


if __name__ == "__main__":
    # run_all_tests()
    result = simulate_channel(
        model="A",
        speed=3.0,
        batch_size=6,
        traj_len=14,
    )
    H = result["H"]
    plot_correlation_curve(H)
    plot_time_series(H)
