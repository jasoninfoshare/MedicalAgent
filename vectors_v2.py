"""
vectors_v2.py - 高级向量数据库模块（支持 Milvus + BM25 + PDF父子检索 + Redis缓存）
医疗健康 AI Agent 项目 V2.0

依赖安装：
    pip install pymilvus langchain-milvus pandas pdfplumber redis
"""

import os
import sys
import json
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
import pandas as pd
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 从 model.py 导入
from model import ZhipuAIEmbeddings, create_embedding_model

# ==================== 配置区 ====================
MILVUS_URI = "./milvus_agent.db"
PDF_MILVUS_URI = "./pdf_agent.db"

# ==================== Redis 基础连接（V1.2 兼容） ====================
try:
    import redis

    def get_redis_client(host='0.0.0.0', port=6379, db=0, password=None, max_connections=10):
        pool = redis.ConnectionPool(
            host=host, port=port, db=db, password=password,
            max_connections=max_connections, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5
        )
        r = redis.StrictRedis(connection_pool=pool)
        try:
            r.ping()
            print("成功连接到 Redis!")
        except redis.ConnectionError:
            print("无法连接到 Redis!")
        return r

    def cache_set(r, question: str, answer: str):
        r.hset("qa", question, answer)
        r.expire("qa", 3600)

    def cache_get(r, question: str):
        return r.hget("qa", question)
except Exception:
    redis = None
    get_redis_client = None
    cache_set = None
    cache_get = None


# ==================== 智谱 Embedding 初始化 ====================
try:
    from zhipuai import ZhipuAI
    ZHIPU_CLIENT = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", ""))
except Exception:
    ZHIPU_CLIENT = None


# ==================== Milvus 向量库（JSONL 数据） ====================
class MilvusVectorStore:
    def __init__(self, client=None, uri: str = MILVUS_URI):
        self.URI = uri
        if client is not None:
            self.embeddings = ZhipuAIEmbeddings(client=client)
        else:
            # 降级使用本地 bce-embedding
            self.embeddings = create_embedding_model()
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }
        self.vectorstore = None

    def create_vector_store(self, docs):
        init_docs = docs[:10]
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            index_params=[self.dense_index, self.sparse_index],
            vector_field=["dense", "sparse"],
            connection_args={"uri": self.URI},
            consistency_level="Bounded",
            drop_old=True,
        )
        print("✅ 已初始化创建 Milvus !")
        count = 10
        temp = []
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            if len(temp) >= 5:
                self.vectorstore.add_documents(temp)
                count += len(temp)
                temp = []
                print(f"已插入 {count} 条数据......")
                time.sleep(0.5)
        if temp:
            self.vectorstore.add_documents(temp)
            count += len(temp)
        print(f"总共插入 {count} 条数据......")
        print("✅ 已创建 Milvus 索引完成 !")
        return self.vectorstore


# ==================== PDF 父子文档检索器 ====================
class PdfRetriever:
    def __init__(self, client=None, uri: str = PDF_MILVUS_URI):
        self.URI = uri
        if client is not None:
            self.embeddings = ZhipuAIEmbeddings(client=client)
        else:
            self.embeddings = create_embedding_model()
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }
        self.docstore = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.retriever = None

    def create_pdf_vector_store(self, docs):
        milvus_vectorstore = Milvus(
            embedding_function=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[self.dense_index, self.sparse_index],
            connection_args={"uri": self.URI},
            consistency_level="Bounded",
            drop_old=True,
        )
        self.retriever = ParentDocumentRetriever(
            vectorstore=milvus_vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        count = 0
        temp = []
        for doc in tqdm(docs):
            temp.append(doc)
            if len(temp) >= 10:
                self.retriever.add_documents(temp)
                count += len(temp)
                temp = []
                print(f"已插入 {count} 条 PDF 数据......")
                time.sleep(0.5)
        if temp:
            self.retriever.add_documents(temp)
            count += len(temp)
        print(f"总共插入 {count} 条 PDF 数据......")
        print("✅ 基于PDF文档数据的 Milvus 索引完成 !")
        return self.retriever


# ==================== 数据预处理 ====================
def prepare_documents(file_path=['./data/data.jsonl', './data/train.jsonl']):
    file_path1 = file_path[0]
    count = 0
    docs = []
    with open(file_path1, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            prompt = content.get('query', '') + "\n" + content.get('response', '')
            temp_doc = Document(page_content=prompt, metadata={"doc_id": str(uuid.uuid4())})
            docs.append(temp_doc)
            count += 1
    print(f"✅ 已加载 {count} 条数据!")
    return docs


def prepare_pdf_document(file_path="./pdf_output/pdf_detailed_text.xlsx"):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['text_content'])
    documents = []
    for _, row in df.iterrows():
        text_content = str(row['text_content']) if pd.notna(row['text_content']) else ""
        doc = Document(
            page_content=text_content.strip(),
            metadata={"doc_id": str(uuid.uuid4())}
        )
        documents.append(doc)
    print(f"成功加载 {len(documents)} 个 PDF 文档")
    return documents


# ==================== 主入口 ====================
if __name__ == "__main__":
    # 示例：构建 JSONL 向量库
    docs = prepare_documents()
    print("预处理文档数据成功......")
    if ZHIPU_CLIENT:
        store = MilvusVectorStore(ZHIPU_CLIENT)
    else:
        store = MilvusVectorStore()
    store.create_vector_store(docs)
    print("全部初始化完成，可以开始问答了......")
