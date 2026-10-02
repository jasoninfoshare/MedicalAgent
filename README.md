# MedicalAgent — a retrieval-augmented Q&A agent over documents and a knowledge graph

**English** | [中文说明](#中文说明)

A personal project exploring how to connect language models to heterogeneous data sources:
dense and sparse retrieval, a Neo4j knowledge graph queried in natural language, and a set of
advanced RAG patterns implemented end to end. Built on LangChain, served through FastAPI.

## What's in here

### Natural language → Cypher (`GraphDatabase/`)

A FastAPI service that translates a natural-language question into a Neo4j Cypher query against a
declared graph schema. Generated queries are not trusted blindly — two validators run before
anything executes:

- `RuleBasedValidator` — structural checks against the schema
- `CypherValidator` — model-based review of the generated query

This is the graph-database counterpart of Text-to-SQL: the hard part is not generating a query, it
is deciding whether the generated query is safe and faithful to the schema before it touches the
database.

### Retrieval pipeline (`vectors.py`, `vectors_v2.py`, `preprocess.py`)

- Dense retrieval over Milvus, sparse BM25, and PDF parent–child retrieval, combined per query
- A batch PDF processor that turns source documents into chunked, embedded records
- Chroma used as a lightweight alternative store for the demo pipelines

### Production concerns (`new_app.py`, `new_redis.py`)

The serving path is where most RAG demos fall over, so this one carries:

- A Redis cache manager addressing Big Key, connection-pool reuse, cache **stampede** (击穿) and
  cache **avalanche** (雪崩)
- Request handling with caching in front of the retrieval and generation path

### Eight retrieval patterns, implemented (`rag_demos/`)

| Pattern | File |
| --- | --- |
| RAG-Fusion — multi-query generation + fused ranking | `rag_fusion_demo.py` |
| Step-back prompting | `step_back_demo.py` |
| Agentic chunking | `agentic_chunker_demo.py` |
| BM25 + vector hybrid retrieval | `bm25_ensemble_demo.py` |
| Contextual compression | `contextual_compression_demo.py` |
| Parent-document retrieval | `parent_document_demo.py` |
| Cross-encoder reranking | `reranker_demo.py` |
| QA generation for retrieval tuning | `qa_generation_demo.py` |

## Services

| Service | Port | Entry point |
| --- | --- | --- |
| Agent (v1) | 8103 | `agent.py` |
| Agent (v2, multi-route recall) | 8103 | `agent_v2.py` |
| Agent (production, Redis-cached) | — | `new_app.py` |
| NL2Cypher | 8101 | `GraphDatabase/main.py` |

## Stack

LangChain · Milvus · Chroma · Neo4j · Redis · FastAPI · BM25 · cross-encoder rerankers ·
OpenAI-compatible and DeepSeek model APIs

> Configuration (API keys, database endpoints) is read from the environment via `.env`; no
> credentials are committed.

---

## 中文说明

一个个人项目,探索如何把大模型接到异构数据源上:稠密与稀疏检索、用自然语言查询的 Neo4j
知识图谱,以及一组完整实现的进阶 RAG 技术。基于 LangChain,通过 FastAPI 提供服务。

### 自然语言转 Cypher(`GraphDatabase/`)

一个 FastAPI 服务,把自然语言问题翻译成针对给定图 schema 的 Neo4j Cypher 查询。
生成的查询不被直接信任,执行前要过两道校验:

- `RuleBasedValidator` —— 基于 schema 的结构校验
- `CypherValidator` —— 模型对生成查询的复核

这是 Text-to-SQL 在图数据库上的对应物。难点不在于生成查询,而在于**在它碰到数据库之前,
判断这条查询是否安全、是否忠实于 schema**。

### 检索链路(`vectors.py`、`vectors_v2.py`、`preprocess.py`)

- Milvus 稠密检索、BM25 稀疏检索、PDF 父子检索,按查询组合
- 批量 PDF 处理器,把源文档转成分块并向量化的记录
- demo 链路里用 Chroma 作为轻量替代存储

### 工程化处理(`new_app.py`、`new_redis.py`)

大多数 RAG demo 都栽在服务链路上,所以这里专门处理了:

- Redis 缓存管理器,针对 Big Key、连接池复用、缓存**击穿**、缓存**雪崩**
- 在检索与生成链路前置缓存的请求处理

### 八种检索技术的完整实现(`rag_demos/`)

| 技术 | 文件 |
| --- | --- |
| RAG-Fusion:多查询生成 + 融合排序 | `rag_fusion_demo.py` |
| Step-back 后退一步提示 | `step_back_demo.py` |
| Agentic 智能分块 | `agentic_chunker_demo.py` |
| BM25 + 向量混合检索 | `bm25_ensemble_demo.py` |
| 上下文压缩 | `contextual_compression_demo.py` |
| 父文档检索 | `parent_document_demo.py` |
| CrossEncoder 重排序 | `reranker_demo.py` |
| 面向检索调优的 QA 生成 | `qa_generation_demo.py` |

### 技术栈

LangChain · Milvus · Chroma · Neo4j · Redis · FastAPI · BM25 · CrossEncoder 重排 ·
OpenAI 兼容接口与 DeepSeek 模型 API

> API key、数据库地址等配置通过 `.env` 从环境读取,仓库中不包含任何凭据。
