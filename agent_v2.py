"""
agent_v2.py - 医疗健康 AI Agent V2.0（多路召回：Milvus稠密+稀疏 + PDF父子检索）
启动方式：python agent_v2.py
接口地址：http://0.0.0.0:8103
"""

import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer, create_embedding_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="医疗 AI Agent V2.0", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 配置区 ====================
URI = "./milvus_agent.db"
URI_PDF = "./pdf_agent.db"

# 尝试使用智谱 Embedding，否则降级本地 bce
try:
    from zhipuai import ZhipuAI
    client_embedding = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", ""))
    embedding_model = ZhipuAIEmbeddings(client_embedding)
    print("✅ 创建 ZhipuAI Embedding 模型成功......")
except Exception:
    embedding_model = create_embedding_model()
    print("✅ 创建本地 bce-embedding 模型成功（降级）......")

# ==================== Milvus 主库（JSONL 数据） ====================
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[
        {"metric_type": "IP", "index_type": "IVF_FLAT"},
        {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}
    ],
    connection_args={"uri": URI},
)
retriever = milvus_vectorstore.as_retriever()
print("✅ 创建 Milvus 连接成功......")

# ==================== PDF 父子文档检索器 ====================
docstore = InMemoryStore()
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pdf_vectorstore = Milvus(
    embedding_function=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    index_params=[
        {"metric_type": "IP", "index_type": "IVF_FLAT"},
        {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}
    ],
    connection_args={"uri": URI_PDF},
    consistency_level="Bounded",
    drop_old=False,
)
parent_retriever = ParentDocumentRetriever(
    vectorstore=pdf_vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
print("✅ 创建 PDF ParentDocumentRetriever 成功......")

# ==================== LLM 客户端 ====================
client_llm = create_deepseek_client()
print("✅ 创建 DeepSeek 客户端成功......")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/")
async def chatbot(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('question')

    # 1. Milvus 主库召回（稠密+稀疏 RRF）
    recall_rerank_milvus = milvus_vectorstore.similarity_search(
        query, k=10, ranker_type="rrf", ranker_params={"k": 100}
    )
    context = format_docs(recall_rerank_milvus) if recall_rerank_milvus else ""

    # 2. PDF 父子文档召回
    retrieved_docs = parent_retriever.invoke(query)
    if retrieved_docs and len(retrieved_docs) >= 1:
        pdf_context = retrieved_docs[0].page_content
        context = context + "\n" + pdf_context

    # 3. 组装 Prompt
    SYSTEM_PROMPT = """
System: 你是一个非常得力的医学助手，你可以通过从数据库中检索出的信息找到问题的答案。
"""
    USER_PROMPT = f"""
User: 利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题，具体的问题介于<question>和</question>之间。如果提供的信息为空，则按照你的经验知识来给出尽可能严谨准确的回答，不知道的时候坦诚的承认不了解，不要编造不真实的信息。
<context>
{context}
</context>
<question>
{query}
</question>
"""

    response = generate_deepseek_answer(client_llm, SYSTEM_PROMPT + USER_PROMPT)
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "response": response,
        "status": 200,
        "time": time_str
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
