import time
import json
import streamlit as st
from agent.react_agent import ReactAgent
import os

st.set_page_config(page_title="5G Helper", layout="wide")

st.title("5G Helper")
st.divider()

# =========================
# Session state 初始化
# =========================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# =========================
# 安全图片路径处理
# =========================
def safe_image_path(path):
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    return path


# =========================
# 渲染函数
# 【修复】render 接收一个可选的容器参数，避免总是创建新的 chat_message 容器
# =========================
def render(role, content, container=None):
    """
    role: "user" | "assistant"
    content: str 或 dict（结构化消息）
    container: 若传入，直接在该容器内渲染；否则创建新的 st.chat_message
    """
    msg = container if container is not None else st.chat_message(role)

    if isinstance(content, dict):
        t = content.get("type")

        if t == "image":
            msg.write(content.get("desc", ""))
            img_path = safe_image_path(content.get("path"))
            if img_path:
                msg.image(img_path, use_container_width=True)
            else:
                msg.error("图片路径不存在")

        elif t == "mixed":
            msg.write(content.get("text", ""))
            img_path = safe_image_path(content.get("image"))
            if img_path:
                msg.image(img_path, use_container_width=True)

        elif t == "error":
            msg.error(content.get("message"))

        else:
            msg.write(str(content))
    else:
        msg.write(content)


# =========================
# 从 full_text 中提取 JSON
# 【修复】流式输出可能包含思考过程、工具调用日志等前缀，
#         不能直接 json.loads，需要找到第一个 '{' 或 '[' 再尝试解析
# =========================
def try_parse_json(text: str):
    """
    尝试从文本末尾提取 JSON 对象/数组。
    返回解析成功的 dict/list，否则返回 None。
    """
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        # 从最后一个结束符向前截取，兼容末尾有换行等情况
        end = text.rfind(end_char)
        if end == -1 or end < start:
            continue
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# =========================
# 历史消息回放
# =========================
for m in st.session_state["messages"]:
    render(m["role"], m["content"])


# =========================
# 用户输入
# =========================
prompt = st.chat_input("请输入问题...")

if prompt:

    # 【修复重复渲染】只 append，不在此处调用 render。
    # 历史循环已经在本次脚本运行时渲染过所有旧消息；
    # 当前新消息通过下方直接创建 chat_message 展示，
    # 下一次脚本重跑时由历史循环统一负责，不会重复。
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 直接在页面上展示当前轮用户消息（不走历史循环）
    with st.chat_message("user"):
        st.write(prompt)

    # ===== Agent 执行 =====
    # 在 spinner 外部提前创建 assistant 容器，
    # 避免 spinner 消失后 chat_message 位置错乱
    assistant_container = st.chat_message("assistant")
    stream_placeholder = assistant_container.empty()

    try:
        with st.spinner("Thinking..."):
            stream = st.session_state["agent"].execute_stream(prompt)

            full_text = ""

            # 流式渲染文本
            for chunk in stream:
                # chunk 可能不是 str（如 None 或其他类型），做防御性处理
                if isinstance(chunk, str):
                    full_text += chunk
                    stream_placeholder.write(full_text + "▌")  # 光标效果
                time.sleep(0.002)

        # 流式结束，清除光标占位符
        stream_placeholder.empty()

        # =========================
        # 最终解析并渲染 assistant 回复
        # =========================
        data = try_parse_json(full_text)

        if isinstance(data, dict) and "type" in data:
            render("assistant", data, container=assistant_container)
            st.session_state["messages"].append({"role": "assistant", "content": data})
        else:
            assistant_container.write(full_text)
            st.session_state["messages"].append({"role": "assistant", "content": full_text})

    except StopIteration:
        # execute_stream 正常耗尽，不应当作错误
        pass

    except Exception as e:
        stream_placeholder.empty()
        err_content = {"type": "error", "message": f"Agent 执行出错：{e}"}
        render("assistant", err_content, container=assistant_container)
        st.session_state["messages"].append({"role": "assistant", "content": err_content})