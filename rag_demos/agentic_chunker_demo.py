"""
rag_demos/agentic_chunker_demo.py - Agentic 智能分块
"""

import os
import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class KnowledgeChunk(BaseModel):
    chunk_title: str = Field(description="这个知识块的简洁明了的标题")
    chunk_text: str = Field(description="从原文中提取并重组的,自包含的文本内容")
    representative_question: str = Field(description="一个可以被这个块内容直接回答的典型问题")


class ChunkList(BaseModel):
    chunks: List[KnowledgeChunk]


parser = PydanticOutputParser(pydantic_object=ChunkList)

prompt_template = """
【角色】:你是一位顶尖的科学文档分析师，你的任务是将复杂的科学文本段落，分解成一组核心的、自包含的"知识块(KnowledgeChunks)"。
【核心任务】:阅读用户提供的文本段落，识别其中包含的独立的核心概念。
【规则】:
1.**自包含性**:每个"知识块"必须是"自包含的(self-contained)"。
2.**概念单一性**:每个"知识块"应该只围绕一个核心概念。
3.**提取并重组**:从原文中提取与该核心概念相关的所有句子，并将它们组合成一个通顺、连贯的段落。
4.**遵循格式**:严格按照下面的JSON格式指令来构建你的输出。
{format_instructions}
【待处理文本】:
{paragraph_text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["paragraph_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

os.environ['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY", "")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
    max_tokens=2048,
).bind(response_format={"type": "json_object"})

chain = prompt | llm | parser


def agentic_chunker(paragraph_text: str) -> List[KnowledgeChunk]:
    try:
        result: ChunkList = chain.invoke({"paragraph_text": paragraph_text})
        return result.chunks
    except Exception:
        return []


document = """
水循环，也称为水文循环，描述了水在地球表面、之上和之下的连续运动。这个循环至关重要，因为它确保了水对所有生命形式的可用性。
循环的第一阶段是蒸发，这是水从海洋、湖泊和河流等表面转化为水蒸气并上升到大气中的过程，植物的蒸腾作用也对此有贡献。
当温暖、潮湿的空气上升并冷却时，会发生第二阶段：凝结。在这个阶段，水蒸气变回微小的液态水滴，形成云。
随着这些水滴碰撞并增长，它们最终变得足够重，以降水的形式落回地球，这是第三阶段，形式可以是雨、雪、雨夹雪或冰雹。
最后，一旦水到达地面，它可能以多种方式移动，构成了第四个阶段：汇集。一些水会作为地表径流流入河流、湖泊和海洋。
其他水则会渗入地下，成为地下水，最终也可能返回地表或海洋，从而重新开始整个循环。
"""

paragraphs = document.strip().split('\n\n')
all_chunks = []
for i, para in enumerate(paragraphs):
    print(f"--- 正在处理第 {i + 1} / {len(paragraphs)} 段 ---")
    chunks_from_para = agentic_chunker(para)
    if chunks_from_para:
        all_chunks.extend(chunks_from_para)

print(f"\n成功从该段落中提取了 {len(all_chunks)} 个知识块.")
for i, chunk in enumerate(all_chunks):
    print(f"【知识块 {i + 1}】")
    print(f"  - 标题: {chunk.chunk_title}")
    print(f"  - 代表性问题: {chunk.representative_question}")
    wrapped_text = textwrap.fill(chunk.chunk_text, width=78, initial_indent='    ', subsequent_indent='    ')
    print(f"  - 文本内容:\n{wrapped_text}")
    print("-" * 80)
