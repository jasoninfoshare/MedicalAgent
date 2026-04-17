"""
rag_demos/bm25_ensemble_demo.py - BM25 + 向量混合检索（EnsembleRetriever）
"""

import os
import uuid
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma

from model import create_embedding_model

embeddings = create_embedding_model()

docs = [
    Document(
        page_content="爸爸前几天去医院刚检查出有糖尿病,今年60岁血糖10请问专家吃什么能好转?医生回复:首先，建议您的父亲遵循医生的建议...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
    Document(
        page_content="不小心得了牛皮癣了，但是不知道有什么特点。感觉和湿疹差不多。牛皮癣和湿疹都是常见的皮肤病...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
]

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

query = "糖尿病患者有什么饮食建议?"
retrieved_docs = ensemble_retriever.invoke(query)
print(f"混合搜索召回了 {len(retrieved_docs)} 个文档。")
for i, doc in enumerate(retrieved_docs):
    print(f"--- 文档 {i+1} ---")
    print(doc.page_content[:200])
