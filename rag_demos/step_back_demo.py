"""
rag_demos/step_back_demo.py - Step-back 后退一步策略
"""

import os
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
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
    Document(page_content="糖尿病饮食建议：控制碳水化合物，增加蔬菜，适量蛋白质，避免高脂高盐...", metadata={"doc_id": str(uuid.uuid4())}),
    Document(page_content="牛皮癣护理：保持皮肤清洁，避免刺激，尽快就医...", metadata={"doc_id": str(uuid.uuid4())}),
]

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

step_back_prompt_template = ChatPromptTemplate.from_messages([
    ("user", "你是一位善于提炼核心问题的专家。请将以下可能很具体的问题,抽象成一个更通用,更高层次的'后退一步'的问题。\n"
             "例如: '有时候头晕,眼前发黑,下肢没有力气,尤其没吃早餐的时候更严重,是什么病?' -> '低血糖的具体症状有哪些?'\n"
             "原始问题: {original_question}")
])
step_back_chain = step_back_prompt_template | llm | StrOutputParser()


def remove_duplicates_by_id(documents):
    seen_ids = set()
    unique_docs = []
    for doc in documents:
        if doc.page_content not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc.page_content)
    return unique_docs


chain = (
    RunnableParallel(
        original_docs=RunnablePassthrough() | retriever,
        step_back_docs=step_back_chain | retriever,
    )
    | RunnableLambda(lambda x: remove_duplicates_by_id(x["original_docs"] + x["step_back_docs"]))
)

user_query = "我有病,不敢吃带糖多的,平时馋了就吃木糖醇的东西,有什么吃饭的好建议吗?"
step_back_docs = chain.invoke(user_query)
print('step_back_docs:', step_back_docs)
