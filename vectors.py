"""
vectors.py - 向量数据库构建、数据入库与索引
医疗健康 AI Agent 项目

使用 Milvus Lite（纯文件，无需安装 Milvus 服务）：
    pip install pymilvus langchain-milvus

数据文件放在 ./data/data.jsonl，每行格式：
    {"query": "问题文本", "response": "回答文本"}

运行方式：
    python vectors.py
"""

import os
import sys
import json
import uuid
import time

# 确保能找到同目录下的 model.py，无论从哪里调用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_milvus.vectorstores.milvus import Milvus as _MilvusBase
from pymilvus import Collection

# 修复 langchain_milvus 0.3.3 + pymilvus 2.6.x 的兼容性 bug：
# MilvusClient 创建的 _using alias 未在 ORM connections 中注册，
# 导致 Collection(..., using=alias) 抛出 ConnectionNotExistException。
_original_extract_fields = _MilvusBase._extract_fields

def _safe_extract_fields(self) -> None:
    from pymilvus import connections
    if not connections.has_connection(self.alias):
        connections.connect(alias=self.alias, **self._connection_args)
    if self.client.has_collection(self.collection_name) and isinstance(self.col, Collection):
        schema = self.col.schema
        for x in schema.fields:
            self.fields.append(x.name)

_MilvusBase._extract_fields = _safe_extract_fields

# 从同目录 model.py 导入
from model import create_embedding_model

# Milvus Lite 数据库文件路径（自动创建，无需启动服务）
MILVUS_URI = "./milvus_agent.db"


# ============================================================
# Milvus Lite 向量库封装
#
# 注意：Milvus Lite 不支持内置 BM25BuiltInFunction，
# 因此这里只使用稠密向量（bce-embedding），
# 关键词召回由上层 BM25Retriever（内存级）补充。
# ============================================================
class MilvusVectorStore:
    """
    基于 Milvus Lite 的稠密向量库。
    检索使用余弦相似度（归一化向量 + 内积 = 余弦）。
    """

    def __init__(self, uri: str = MILVUS_URI):
        self.uri = uri
        self.embeddings = create_embedding_model()
        self.vectorstore: Milvus | None = None

    def create_vector_store(self, docs: list[Document]) -> Milvus:
        """
        批量构建向量索引并写入 Milvus Lite。
        一次性写入全部文档（由 langchain-milvus 内部自动分批 embedding）。
        """
        print(f"共 {len(docs)} 条文档，开始写入 Milvus Lite...")

        self.vectorstore = Milvus.from_documents(
            documents=docs,
            embedding=self.embeddings,
            connection_args={"uri": self.uri},
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            drop_old=True,
        )
        print(f"✅ 总共写入 {len(docs)} 条，Milvus Lite 索引构建完成")
        return self.vectorstore


# ============================================================
# 数据预处理：从 JSONL 文件加载文档
# ============================================================
def prepare_documents(
    file_paths=("./data/data.jsonl",),
) -> list[Document]:
    """
    读取 JSONL 文件，每行 {"query": ..., "response": ...}，
    拼接为 LangChain Document 列表。
    """
    docs: list[Document] = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"[SKIP] 文件不存在：{file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                content = json.loads(line)
                text = content.get("query", "") + "\n" + content.get("response", "")
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"doc_id": str(uuid.uuid4())},
                    )
                )
        print(f"  已加载 {len(docs)} 条（来自 {file_path}）")
    return docs


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 1. 加载数据
    docs = prepare_documents(["./data/data.jsonl", "./data/train.jsonl"])
    if not docs:
        print("[ERROR] 没有加载到任何文档，请检查 ./data/ 目录下的 jsonl 文件")
        exit(1)
    print(f"✅ 预处理完成，共 {len(docs)} 条文档")

    # 2. 构建向量库
    store = MilvusVectorStore()
    store.create_vector_store(docs)
    print(f"✅ 向量库已保存到 {MILVUS_URI}，可以启动 agent.py 了")