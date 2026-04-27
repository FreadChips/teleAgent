# sionna_tools.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.path_tool import get_abs_path
from utils.config_handler import agent_conf

# ====== 全局缓存（避免重复初始化） ======
_CTX = {}

def _init():
    global _CTX
    if _CTX:
        return _CTX

    from sionna.channel import GenerateOFDMChannel
    from sionna.channel.tr38901 import CDL, AntennaArray
    from sionna.ofdm import ResourceGrid

    # 避免显存爆
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(0)

    def build_rg(traj_len, fft_size):
        return ResourceGrid(
            num_ofdm_symbols=traj_len,
            fft_size=fft_size,
            subcarrier_spacing=15e3,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=20,
        )

    def build_ant(n, fc):
        return AntennaArray(
            num_rows=1,
            num_cols=n,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=fc,
        )

    _CTX = {
        "GenerateOFDMChannel": GenerateOFDMChannel,
        "CDL": CDL,
        "build_rg": build_rg,
        "build_ant": build_ant,
    }
    return _CTX


def simulate_channel(
    model="A",
    speed=3.0,
    delay_spread=100e-9,
    carrier_frequency=3.5e9,
    batch_size=8,
    traj_len=14,
    fft_size=128,
):
    """
    返回：
        dict {
            "H": complex ndarray [B, T, FFT]
        }
    """
    ctx = _init()

    rg = ctx["build_rg"](traj_len, fft_size)

    cdl = ctx["CDL"](
        model=model,
        delay_spread=delay_spread,
        carrier_frequency=carrier_frequency,
        ut_array=ctx["build_ant"](1, carrier_frequency),
        bs_array=ctx["build_ant"](1, carrier_frequency),
        direction="uplink",
        min_speed=speed,
        max_speed=speed,
    )

    gen = ctx["GenerateOFDMChannel"](
        channel_model=cdl,
        resource_grid=rg,
        normalize_channel=True,
    )

    h = gen(batch_size).numpy()
    h = h[:, 0, 0, 0, 0, :, :]  # [B, T, FFT]

    return {"H": h}


def compute_time_correlation(H):
    """
    输入：
        H: [B, T, FFT]
    返回：
        dict
    """
    sc = H.shape[-1] // 2
    h_sc = H[:, :, sc]

    def corr(x, y):
        return np.real(
            np.mean(x * np.conj(y))
            / np.sqrt(np.mean(np.abs(x) ** 2) * np.mean(np.abs(y) ** 2))
        )

    corrs = []
    for i in range(H.shape[0]):
        c = corr(h_sc[i, :-1], h_sc[i, 1:])
        corrs.append(float(c))

    return {
        "mean_corr": float(np.mean(corrs)),
        "corr_list": corrs,
    }

def plot_time_series(
    H,
    save_path,
):
    """
    绘制单个子载波随时间变化（幅度）

    输入:
        H: [B, T, FFT]
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sc = H.shape[-1] // 2
    h_sc = H[0, :, sc]   # 取第一条轨迹

    amp = np.abs(h_sc)

    plt.figure()
    plt.plot(amp)
    plt.title("Channel Amplitude vs Time")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_correlation_curve(
    H,
    save_path,
):
    """
    绘制时间相关性曲线（lag相关）
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sc = H.shape[-1] // 2
    h_sc = H[0, :, sc]

    def corr(x, y):
        return np.real(
            np.mean(x * np.conj(y))
            / np.sqrt(np.mean(np.abs(x) ** 2) * np.mean(np.abs(y) ** 2))
        )

    max_lag = min(10, len(h_sc) - 1)
    corrs = []

    for lag in range(1, max_lag + 1):
        c = corr(h_sc[:-lag], h_sc[lag:])
        corrs.append(c)

    plt.figure()
    plt.plot(range(1, max_lag + 1), corrs)
    plt.title("Time Correlation vs Lag")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path