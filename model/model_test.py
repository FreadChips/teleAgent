from langchain_ollama import OllamaEmbeddings
import numpy as np

# 初始化
embed_model = OllamaEmbeddings(model="bge-m3")


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 定义三组词汇
text1 = "Orthogonal Frequency Division Multiplexing" # OFDM全称
text2 = "OFDM modulation in 5G NR"                # 相关联
text3 = "5G技术"                      # 完全无关

v1 = embed_model.embed_query(text1)
v2 = embed_model.embed_query(text2)
v3 = embed_model.embed_query(text3)

print(f"相关词相似度: {cosine_similarity(v1, v2):.4f}")
print(f"无关词相似度: {cosine_similarity(v2, v3):.4f}")