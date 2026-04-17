"""
rag_demos/qa_generation_demo.py - QA生成优化：为文档块生成代理问题，提升召回率
"""

import os
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain.retrievers import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from model import create_embedding_model, create_deepseek_client

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
        page_content="爸爸前几天去医院刚检查出有糖尿病,今年60岁血糖10请问专家吃什么能好转?\n医生回复:首先，建议您的父亲遵循医生的建议，按时服药和定期检查血糖。其次，饮食对于糖尿病患者非常重要...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
    Document(
        page_content="不小心得了牛皮癣了，但是不知道有什么特点。感觉和湿疹差不多。\n牛皮癣和湿疹都是常见的皮肤病，但两者的病因和治疗方法不同...",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
]

doc_ids = [doc.metadata["doc_id"] for doc in docs]

question_gen_prompt_str = (
    "你是一位AI医学专家。请根据以下文档内容,生成3个用户可能会提出的,高度相关的问题。\n"
    "只返回问题列表，每个问题占一行，不要有其他前缀或编号。\n"
    "文档内容:\n----------\n{content}\n----------\n"
)

question_gen_prompt = ChatPromptTemplate.from_template(question_gen_prompt_str)
question_generator_chain = question_gen_prompt | llm | StrOutputParser()

sub_docs = []
for i, doc in enumerate(docs):
    doc_id = doc_ids[i]
    generated_questions = question_generator_chain.invoke({"content": doc.page_content}).split("\n")
    generated_questions = [q.strip() for q in generated_questions if q.strip()]
    for q in generated_questions:
        sub_docs.append(Document(page_content=q, metadata={"doc_id": doc_id}))

print("创建Chroma向量数据库, 并添加文档...")
vectorstore_qa = Chroma.from_documents(documents=sub_docs, embedding=embeddings)
doc_store = InMemoryStore()
doc_store.mset(list(zip(doc_ids, docs)))

multivector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore_qa,
    docstore=doc_store,
    id_key="doc_id",
)

user_query = "糖尿病患者有什么饮食建议?"
retrieved_qa_docs = multivector_retriever.invoke(user_query)
print('检索结果:', retrieved_qa_docs)
