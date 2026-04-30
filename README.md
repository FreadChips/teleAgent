# teleAgent — 5G/6G 无线通信 AI 研究助手

基于 LLM + NVIDIA Sionna 的领域专用 AI 研究助手，面向 5G/6G 物理层通信研究场景，在对话中完成 **理论检索 → 仿真实验 → 可视化分析 → 结论输出** 的完整研究闭环。

## 核心能力

| 能力 | 触发场景 | 实现方式 |
|---|---|---|
| **A - 理论检索 (RAG)** | 3GPP 协议、通信原理、公式推导类问题 | ChromaDB 向量检索 + LLM 摘要 |
| **B - 单点仿真** | 特定条件下的信道状态查询 | Sionna CDL 信道模型实时仿真 |
| **C - 对比实验** | 不同参数（速度、模型）对信道的影响 | 双条件并行仿真 + 数据对比 |
| **D - 可视化绘图** | 波形、时域曲线、相关函数图 | Matplotlib 生成 PNG 图表 |
| **E - 闲聊** | 问候语、总结归纳 | LLM 直接回复，不调用工具 |

## 技术架构

```
app.py (Streamlit 前端)
    │
    ▼
agent/ (ReAct Agent 核心)
    ├── react_agent.py      LangGraph ReAct 循环编排
    ├── tools/
    │   ├── agent_tools.py   5 个 LangChain @tool 工具定义
    │   ├── sionna_tools.py  Sionna 信道仿真引擎 + 绘图
    │   └── middleware.py    LangGraph 中间件（工具监控 + 模型调用日志）
    │
    ▼
rag/ (检索增强生成)
    ├── vector_store.py     ChromaDB 向量库（MD5 去重 / 批量导入）
    └── rag_service.py     检索 + 摘要服务
    │
    ▼
model/ (模型工厂)
    └── factory.py          ChatTongyi (qwen3-max) + OllamaEmbeddings (bge-m3)
```

## 技术栈

| 类别 | 技术 | 用途 |
|---|---|---|
| LLM 框架 | LangChain + LangGraph | Agent 构建、工具定义、链式调用 |
| 前端 | Streamlit | Web 聊天界面 |
| 对话模型 | 阿里通义千问 Qwen3-Max | 对话推理 |
| 嵌入模型 | BGE-M3 (Ollama 本地) | 文档向量化 |
| 向量数据库 | ChromaDB | 知识库存储与检索 |
| 信道仿真 | NVIDIA Sionna + TensorFlow | 3GPP TR 38.901 CDL 信道模型 |
| 数值计算 | NumPy | 相关性及统计分析 |
| 可视化 | Matplotlib | 仿真结果绘图 |
| 文档解析 | PyMuPDF | PDF 知识库加载 |

## 目录结构

```
teleagentv1/
  app.py                  Streamlit 入口（"5G Helper" 聊天界面）
  teleagent.md            项目文档（中文）

  agent/                  Agent 核心
    react_agent.py          ReAct Agent 主类
    tools/
      agent_tools.py          LangChain 工具定义
      sionna_tools.py         Sionna 信道仿真引擎
      middleware.py           LangGraph 中间件

  config/                 YAML 配置文件
    agent.yml                Agent / 高德 API 配置
    chroma.yml               向量库配置
    prompts.yml              提示词路径配置
    rag.yml                  RAG 模型配置

  prompts/                提示词模板
    main_prompt.txt          ReAct Agent 系统提示词
    rag_summarize.txt        RAG 摘要提示词

  rag/                    RAG 检索增强
    rag_service.py           检索 + 摘要服务
    vector_store.py          ChromaDB 向量库管理

  model/                  模型工厂
    factory.py               ChatModelFactory + EmbeddingsFactory

  utils/                  工具模块
    config_handler.py        YAML 配置加载
    file_handler.py          PDF/TXT 加载、MD5 计算
    logger_handler.py        双通道日志（控制台 + 按日滚动文件）
    path_tool.py             项目根路径解析
    prompt_loader.py         提示词模板读取

  chroma_db/              ChromaDB 持久化数据
  data/                   知识库文档目录
  result/                 仿真结果图片输出
  logs/                   运行日志输出
```

## 快速开始

### 环境要求

- Python 3.10+
- Ollama（本地运行 BGE-M3 嵌入模型）
- 聊天模型 API Key

### 安装步骤

```bash
# 1. 安装依赖
pip install streamlit langchain langgraph chromadb pymupdf pyyaml numpy matplotlib tensorflow

# 2. 安装 NVIDIA Sionna
pip install sionna

# 3. 拉取嵌入模型
ollama pull bge-m3

# 4. 配置 API Key
# 编辑 config/rag.yml，填入 API Key

# 5. 启动应用
streamlit run app.py
```

### 知识库导入

将 3GPP 标准文档（PDF/TXT）放入 `data/` 目录，系统启动时会自动导入并建立向量索引（MD5 去重，支持增量更新）。

## 设计模式

- **ReAct Agent** — Think → Act → Observe 推理循环，基于 LangGraph 实现
- **抽象工厂** — `BaseModelFactory` → `ChatModelFactory` / `EmbeddingsFactory`
- **单例模式** — 模块级 `chat_model` / `embed_model` 实例
- **中间件/拦截器** — `@wrap_tool_call` / `@before_model` 装饰器
- **配置驱动** — 所有可调参数外置到 YAML 文件


