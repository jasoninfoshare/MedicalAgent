"""
rag_demos/reranker_demo.py - 重排序（CrossEncoderReranker）优化检索结果
"""

import os
import uuid
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="maidalun1020/bce-embedding-base_v1",
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    model_kwargs={"local_files_only": True},
)

docs = [
    Document(
        page_content="爸爸前几天去医院刚检查出有糖尿病,今年60岁血糖10请问专家吃什么能好转?医生回复:首先饮食控制...",
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

model = HuggingFaceCrossEncoder(model_name="maidalun1020/bce-reranker-base_v1", model_kwargs={"local_files_only": True})
compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

compressed_docs = compression_retriever.invoke("湿疹和什么疾病症状很相近?")
print("重排序后结果:")
for i, doc in enumerate(compressed_docs):
    print(f"{i+1}. {doc.page_content[:200]}")
