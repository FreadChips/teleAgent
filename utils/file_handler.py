import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

def get_file_md5_hex(filepath: str):     # 获取文件的md5的十六进制字符串

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096       # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)


            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None




def pdf_loader(filepath: str) -> list[Document]:
    """
    使用 PyMuPDF 进行 PDF 解析。
    PyMuPDF 对公式和特殊字符的兼容性更好，速度也更快。
    注意：PyMuPDFLoader 默认不直接支持在构造函数传 passwd，
    如果确实有加密文档，通常建议先解密或参考 PyMuPDF 官方文档。
    """
    try:
        # PyMuPDFLoader 会自动处理文档读取
        loader = PyMuPDFLoader(filepath)
        return loader.load()
    except Exception as e:
        logger.error(f"PyMuPDF 解析文件 {filepath} 失败: {str(e)}")
        return []


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
