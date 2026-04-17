"""
rag_demos/parent_document_demo.py - 父文档检索器（ParentDocumentRetriever）
"""

import uuid
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

from model import create_embedding_model

embeddings = create_embedding_model()

docs = [
    Document(
        page_content="爸爸前几天去医院刚检查出有糖尿病...（此处省略长文本）...医生建议控制饮食、定期复查。",
        metadata={"doc_id": str(uuid.uuid4())}
    ),
]

vectorstore = Chroma(embedding_function=embeddings, collection_name="split_parents")
store = InMemoryStore()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
child_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=10)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)

query = "糖尿病患者有什么饮食建议?"
retrieved_docs = retriever.invoke(query)
print('父文档检索结果:', retrieved_docs)
for doc in retrieved_docs:
    print("-", doc.page_content[:300])
