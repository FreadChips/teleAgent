from langchain_chroma import Chroma
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        base_data_path = get_abs_path(chroma_conf["data_path"])
        allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
        allowed_files_path = []

        for root, dirs, files in os.walk(base_data_path):
            for file in files:
                if file.endswith(allowed_types):
                    # 获取文件的完整绝对路径
                    allowed_files_path.append(os.path.join(root, file))

        for path in allowed_files_path:

            # # 因为文档太多，测试用其中几个文档
            # print(path)
            # cur_folder = (os.path.basename(os.path.dirname(path)))
            # print(cur_folder)
            # if cur_folder != "paper":
            #     continue
            print(path)
            # ==============

            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_docs: list[Document] = self.spliter.split_documents(documents)
                # --- 增加清洗逻辑 ---
                cleaned_docs = []
                for doc in split_docs:
                    content = doc.page_content.strip()
                    # 剔除字数太少的（可能是解析错误的页码或公式残片）
                    # 剔除只有特殊符号的
                    if len(content) > 10:  # 阈值可以根据需要调整
                        cleaned_docs.append(doc)
                    else:
                        logger.warning(f"跳过无效分片: {content}")

                split_docs = cleaned_docs

                if not split_docs:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue


                # 提取文件所在的直接父文件夹名称（例如：book, standard, paper）
                category = os.path.basename(os.path.dirname(path))

                for doc in split_docs:
                    # 确保 metadata 字典存在
                    if doc.metadata is None:
                        doc.metadata = {}
                    # 将类别名称作为 metadata 写入，方便检索时进行过滤
                    doc.metadata["category"] = category
                    doc.metadata["source"] = path  # 保留原文件路径溯源

                batch_size = 100
                for i in range(0, len(split_docs), batch_size):
                    batch = split_docs[i: i + batch_size]
                    try:
                        self.vector_store.add_documents(batch)
                        logger.info(f"已完成 {path} 的第 {i // batch_size + 1} 批次分片存储")
                    except Exception as e:
                        # 如果这一批次失败了，尝试单条插入，找出“毒瘤”
                        logger.warning(f"Batch 写入失败，切换为单条模式排查...")
                        for single_doc in batch:
                            try:
                                self.vector_store.add_documents([single_doc])
                            except:
                                logger.error(f"跳过无法处理的分片: {single_doc.page_content[:50]}...")
                                continue


                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("gold smith")
    for r in res:
        print(r.page_content)
        print("-"*20)


