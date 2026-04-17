"""
rag_demos/rag_fusion_demo.py - RAG-Fusion：多查询生成 + RRF 倒数排序融合
"""

import os
import uuid
import operator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    Document(
        page_content="爸爸前几天去医院刚检查出有糖尿病...饮食控制非常重要...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
    Document(
        page_content="不小心得了牛皮癣了...牛皮癣和湿疹都是常见的皮肤病...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
]

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

query_gen_prompt = ChatPromptTemplate.from_messages([
    ("user", "你是一位AI医学专家。请根据以下问题,生成3个不同角度的,语义相似的查询。\n"
             "每个查询占一行,不要有其他前缀或编号。\n原始问题: {original_question}")
])
generate_queries_chain = query_gen_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))


def reciprocal_rank_fusion(retrieval_results, k=60):
    fused_scores = {}
    for doc_list in retrieval_results:
        for rank, doc in enumerate(doc_list):
            doc_id = doc.page_content
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank)
    reranked_results = [
        next((doc for doc_list in retrieval_results for doc in doc_list if doc.page_content == doc_id), None)
        for doc_id, score in sorted(fused_scores.items(), key=operator.itemgetter(1), reverse=True)
    ]
    return [doc for doc in reranked_results if doc is not None]


def rag_fusion_pipeline(original_question: str):
    generated_queries = generate_queries_chain.invoke({"original_question": original_question})
    all_queries = [original_question] + [q for q in generated_queries if q.strip()]
    print(f"生成的查询: {all_queries}")
    retriever = vectorstore.as_retriever()
    retrieval_results = [retriever.invoke(q) for q in all_queries]
    final_docs = reciprocal_rank_fusion(retrieval_results)
    return final_docs


user_query = "糖尿病患者有什么饮食建议?"
fusion_docs = rag_fusion_pipeline(user_query)
print(f"RAG-Fusion 检索结果 ({len(fusion_docs)} 个):")
for doc in fusion_docs:
    print("-", doc.page_content[:150])
