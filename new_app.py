"""
new_app.py - 生产级 Agent 服务（集成 Redis 缓存 + 防击穿/防雪崩）
启动方式：python new_app.py
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

from model import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer, create_embedding_model
from new_redis import redis_manager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="医疗 AI Agent V2.1", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Embedding 模型 ====================
try:
    from zhipuai import ZhipuAI
    client_embedding = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", ""))
    embedding_model = ZhipuAIEmbeddings(client_embedding)
    print("✅ 创建 ZhipuAI Embedding 模型成功......")
except Exception:
    embedding_model = create_embedding_model()
    print("✅ 创建本地 bce-embedding 模型成功（降级）......")

# ==================== Milvus 向量库 ====================
URI = "./milvus_agent.db"
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

# ==================== LLM 客户端 ====================
client_llm = create_deepseek_client()
print("✅ 创建 DeepSeek 客户端成功......")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def perform_rag_and_llm(query: str) -> str:
    recall_milvus = milvus_vectorstore.similarity_search(
        query, k=10, ranker_type="rrf", ranker_params={"k": 100}
    )
    context = format_docs(recall_milvus) if recall_milvus else ""

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
    return generate_deepseek_answer(client_llm, SYSTEM_PROMPT + USER_PROMPT)


@app.post("/")
async def chatbot(request: Request):
    try:
        json_post_raw = await request.json()
        if isinstance(json_post_raw, str):
            json_post_list = json.loads(json_post_raw)
        else:
            json_post_list = json_post_raw
        query = json_post_list.get('question')
        if not query:
            return {"status": 400, "error": "Question is required"}

        compute_callback = lambda: perform_rag_and_llm(query)
        response = redis_manager.get_or_compute(query, compute_callback)
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "response": response,
            "status": 200,
            "time": time_str
        }
    except Exception as e:
        print(f"Server Error: {e}")
        return {"status": 500, "error": str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
