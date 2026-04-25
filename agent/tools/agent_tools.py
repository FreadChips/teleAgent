# agent_tools.py

import os
import json
import random
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from langchain_core.tools import tool

from utils.logger_handler import logger
from utils.config_handler import agent_conf
from rag.rag_service import RagSummarizeService

# ====== 引入 Sionna ======
from sionna_tools import simulate_channel, compute_time_correlation, plot_time_series, plot_correlation_curve


rag = RagSummarizeService()

# =========================================================
# 基础工具（你原有的）
# =========================================================

_IPV4_RE = re.compile(
    r"^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
    r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)

def _is_valid_ipv4(ip: str) -> bool:
    return bool(_IPV4_RE.match(ip or ""))


def _get_public_ip() -> str:
    ip_sources = agent_conf.get("public_ip_sources", [
        "https://ipv4.icanhazip.com",
    ])
    timeout = float(agent_conf.get("public_ip_timeout", 3))

    for source in ip_sources:
        try:
            with urlopen(source, timeout=timeout) as resp:
                ip = resp.read().decode("utf-8").strip()
                if _is_valid_ipv4(ip):
                    return ip
        except Exception:
            continue
    return ""


GAODE_BASE_URL = agent_conf.get("gaode_base_url")
GAODE_TIMEOUT = float(agent_conf.get("gaode_timeout"))


def _gaode_get(path: str, params: dict) -> dict:
    gaode_key = (agent_conf.get("gaodekey") or "").strip()
    if not gaode_key:
        raise ValueError("agent.yml中未配置gaodekey")

    query = dict(params)
    query["key"] = gaode_key
    url = f"{GAODE_BASE_URL}{path}?{urlencode(query)}"

    with urlopen(url, timeout=GAODE_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# =========================================================
# Tool 1：RAG
# =========================================================

@tool(description="从向量数据库检索通信知识")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


# =========================================================
# Tool 2：Sionna信道仿真（核心）
# =========================================================

@tool(description="使用Sionna生成信道并返回时间相关性")
def channel_simulator(
    model: str = "A",
    speed: float = 3.0,
) -> str:
    """
    返回信道统计结果（轻量）
    """
    try:
        result = simulate_channel(
            model=model,
            speed=speed,
            batch_size=6,
            traj_len=14,
        )

        stats = compute_time_correlation(result["H"])

        return json.dumps({
            "model": model,
            "speed": speed,
            "mean_time_corr": stats["mean_corr"],
            "sample_corr": stats["corr_list"][:3],
        })

    except Exception as e:
        logger.error(f"[channel_simulator] err={str(e)}")
        return "信道仿真失败"


# =========================================================
# Tool 3：对比实验（Agent推理增强）
# =========================================================

@tool(description="对比不同速度下信道时间相关性")
def channel_compare(
    speed_low: float = 3.0,
    speed_high: float = 30.0,
) -> str:

    r1 = channel_simulator.invoke({
        "model": "A",
        "speed": speed_low
    })

    r2 = channel_simulator.invoke({
        "model": "A",
        "speed": speed_high
    })

    return f"""
低速 ({speed_low} m/s):
{r1}

高速 ({speed_high} m/s):
{r2}
"""

@tool(description="生成信道幅度随时间变化图")
def channel_plot_time(
    model: str = "A",
    speed: float = 3.0,
) -> str:

    try:
        from sionna_tools import simulate_channel, plot_time_series

        result = simulate_channel(
            model=model,
            speed=speed,
            batch_size=1,
            traj_len=30,
        )

        path = plot_time_series(result["H"])

        return f"已生成信道时间幅度图: {path}"

    except Exception as e:
        return f"绘图失败: {str(e)}"

@tool(description="生成信道时间相关性曲线图")
def channel_plot_correlation(
    model: str = "A",
    speed: float = 3.0,
) -> str:

    try:
        from sionna_tools import simulate_channel, plot_correlation_curve

        result = simulate_channel(
            model=model,
            speed=speed,
            batch_size=1,
            traj_len=30,
        )

        path = plot_correlation_curve(result["H"])

        return f"已生成信道相关性曲线: {path}"

    except Exception as e:
        return f"绘图失败: {str(e)}"



if __name__ == '__main__':

    print(0)