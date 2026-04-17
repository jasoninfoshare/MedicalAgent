"""
agent.py - Agent 核心处理逻辑 + FastAPI 服务
医疗健康 AI Agent 项目

启动方式：
    python agent.py

接口地址：http://0.0.0.0:8103
请求示例：
    POST /
    Content-Type: application/json
    {"question": "糖尿病患者有什么饮食建议？"}
"""

import os
import sys
import json
import datetime

# 确保能找到同目录下的 model.py，无论从哪里调用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_milvus import Milvus
from langchain_milvus.vectorstores.milvus import Milvus as _MilvusBase
from pymilvus import Collection

# 修复 langchain_milvus 0.3.3 + pymilvus 2.6.x 兼容性：
# MilvusClient 创建的 _using alias 未在 ORM connections 中注册，
# 导致 Collection(..., using=alias) 抛出 ConnectionNotExistException。

def _safe_extract_fields(self) -> None:
    from pymilvus import connections
    if not connections.has_connection(self.alias):
        connections.connect(alias=self.alias, **self._connection_args)
    if self.client.has_collection(self.collection_name) and isinstance(self.col, Collection):
        schema = self.col.schema
        for x in schema.fields:
            self.fields.append(x.name)

_MilvusBase._extract_fields = _safe_extract_fields

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# 从同目录 model.py 导入
from model import create_embedding_model, create_deepseek_client, generate_deepseek_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MILVUS_URI = "./milvus_agent.db"   # vectors.py 生成的数据库文件

# ============================================================
# FastAPI 初始化
# ============================================================
app = FastAPI(title="医疗 AI Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 服务启动时初始化模型 & 检索器
# ============================================================
print("加载 bce-embedding 模型...")
embedding_model = create_embedding_model()
print("✅ Embedding 模型加载完成")

print("连接 Milvus Lite...")
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": MILVUS_URI},
)
# 稠密向量检索器（语义相似度，召回 top-5）
dense_retriever = milvus_vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ Milvus Lite 连接成功")

# BM25 稀疏检索器（关键词匹配）
# 从 Milvus 中取出所有文档，初始化内存级 BM25
print("初始化 BM25 检索器...")
_all_docs = milvus_vectorstore.similarity_search("", k=9999)   # 拉取全量文档
if _all_docs:
    bm25_retriever = BM25Retriever.from_documents(_all_docs)
    bm25_retriever.k = 5
    # 混合检索：BM25(0.4) + Dense(0.6)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    print("✅ 混合检索器（BM25 + Dense）初始化成功")
else:
    # 数据库为空时降级为纯稠密检索
    retriever = dense_retriever
    print("⚠️  数据库为空，使用纯稠密向量检索（建议先运行 vectors.py 入库）")

print("连接 DeepSeek API...")
client_llm = create_deepseek_client()
print("✅ DeepSeek 客户端创建成功")


# ============================================================
# 工具函数
# ============================================================
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(context: str, query: str) -> str:
    system = (
        "System: 你是一个非常得力的医学助手，"
        "你可以通过从数据库中检索出的信息找到问题的答案。\n\n"
    )
    user = (
        "User: 利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题，"
        "具体的问题介于<question>和</question>之间。"
        "如果提供的信息为空，则按照你的经验知识来给出尽可能严谨准确的回答，"
        "不知道的时候坦诚地承认不了解，不要编造不真实的信息。\n"
        f"<context>\n{context}\n</context>\n"
        f"<question>\n{query}\n</question>\n"
    )
    return system + user


# ============================================================
# API 路由
# ============================================================
@app.post("/")
async def chatbot(request: Request):
    """
    输入：{"question": "用户问题"}
    输出：{"response": "...", "status": 200, "time": "..."}
    """
    body = await request.json()
    query: str = body.get("question", "").strip()

    if not query:
        return {"response": "请提供问题", "status": 400, "time": ""}

    # 混合检索（BM25 + 稠密向量，EnsembleRetriever 内部做 RRF 融合）
    recalled_docs = retriever.invoke(query)
    context = format_docs(recalled_docs) if recalled_docs else ""

    # 构建 Prompt 并调用 MiniMax 生成回答
    prompt = build_prompt(context, query)
    response_text = generate_deepseek_answer(client_llm, prompt)

    now = datetime.datetime.now()
    return {
        "response": response_text,
        "status": 200,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

# 挂载静态文件（前端页面），需放在 API 路由之后，使 POST / 仍由 chatbot 处理
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103, workers=1)