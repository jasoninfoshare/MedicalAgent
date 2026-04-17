"""
rag_demos/contextual_compression_demo.py - 上下文压缩（ContextualCompressionRetriever）
"""

import os
import uuid
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from model import create_embedding_model

os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY", "")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=2048,
)

embeddings = create_embedding_model()

docs = [
    Document(page_content="糖尿病饮食：控制碳水化合物，增加蔬菜，适量蛋白质，避免高脂高盐，规律三餐，定期运动...", metadata={"doc_id": str(uuid.uuid4())}),
    Document(page_content="牛皮癣护理：保持皮肤清洁，避免刺激，尽快就医，医生会根据病情给出外用药、口服药、光疗等方案...", metadata={"doc_id": str(uuid.uuid4())}),
]

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

compressor = LLMChainExtractor.from_llm(llm=llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

query = "糖尿病患者有什么饮食建议?"
retrieved_compressed_docs = compression_retriever.invoke(query)
print(f"上下文压缩检索结果:")
for i, doc in enumerate(retrieved_compressed_docs):
    original_len = len(doc.metadata.get('original_content', doc.page_content))
    compressed_len = len(doc.page_content)
    print(f"文档 {i+1}(原始长度参考: {original_len}, 压缩后长度: {compressed_len}):")
    print(doc.page_content)
    print("-" * 30)
