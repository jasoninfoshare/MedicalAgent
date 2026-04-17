# NLP2SQL & AI销售顾问 项目笔记

> 整理自开发过程中的实际问题与解决方案

---

## 项目一：NLP2SQL

### 项目目标
用自然语言问问题，自动生成 SQL 查询数据库，返回结果。

### 核心流程
```
用户说"前五名艺术家是谁"
        ↓
LLM 理解意图，生成 SQL
SELECT ar.Name, COUNT(*) FROM tracks JOIN artists...
        ↓
拿 SQL 去真实数据库执行
        ↓
返回结果 [('Iron Maiden', 213), ('U2', 135)...]
```

### 关键技术：Function Calling
不是让 LLM 直接回答，而是告诉 LLM「你有一个工具叫 `ask_database`，需要查数据时就调用它生成 SQL」。LLM 决定什么时候调用、生成什么 SQL，代码负责真正去执行这条 SQL。

### 文件结构
```
nlp2sql.py          # 主程序（4步合一）
requirements.txt    # 依赖
Chinook_Sqlite.sqlite  # 数据库文件（需手动下载）
```

### 配置说明
```python
# 使用 MiniMax 替换 OpenAI
MINIMAX_API_KEY  = "your-key"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
GPT_MODEL        = "MiniMax-M2.5"
```

### 遇到的坑

#### 1. 数据库文件损坏
**问题：** `sqlite3.DatabaseError: file is not a database`

**原因：** GitHub release 链接下载到的是 HTML 重定向页而非真实文件

**解决：** 手动下载 `Chinook_Sqlite.sqlite` 放到项目目录，修改代码路径：
```python
DB_PATH = "./Chinook_Sqlite.sqlite"
```

---

## 项目二：AI销售顾问（RAG）

### 项目目标
让 LLM 基于私有知识库回答问题，而不是靠自身训练数据瞎编。

### 核心架构（RAG）

```
【阶段一：建库，只跑一次】
读取销售话术文本
        ↓
切割成70个小块（每块一问一答）
        ↓
每个块用 Embedding 模型转成向量
        ↓
存入 Chroma 向量数据库（本地 sales_assistant_db/）

【阶段二：问答】
用户问"小区吵不吵？"
        ↓
把问题也转成向量
        ↓
在数据库里找最相似的2~3条话术（语义检索）
        ↓
把检索到的话术 + 用户问题一起发给 LLM
        ↓
LLM 基于真实话术生成回答
        ↓
Gradio 展示对话界面
```

### 文件结构
```
demo2_build_db.py          # 建库脚本（只跑一次）
demo2_app.py               # Gradio 应用
real_estate_sales_data.txt # 私域知识库（销售话术）
requirements2.txt          # 依赖
sales_assistant_db/        # 自动生成的向量数据库目录
```

### 运行顺序
```bash
pip install -r requirements2.txt
pip install sentence-transformers   # Embedding 依赖

# 首次运行，建立向量数据库
export HF_ENDPOINT=https://hf-mirror.com  # 国内加速下载模型
python demo2_build_db.py

# 启动应用
python demo2_app.py
```

### 配置说明
```python
# LLM 使用 MiniMax
MINIMAX_API_KEY  = "your-key"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
LLM_MODEL        = "MiniMax-M2.5"

# Embedding 使用本地免费模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文优化，约100MB，首次自动下载
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

### 遇到的坑

#### 1. langchain-classic 不存在
**问题：** `ERROR: No matching distribution found for langchain-classic`

**原因：** PDF 中包名写错了

**解决：**
```python
# 错误
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
# 正确
from langchain.chains import RetrievalQA
```

#### 2. MiniMax Embedding 参数错误
**问题：** `KeyError: 'data'`

**原因：** MiniMax embedding 接口的参数名和返回格式与 OpenAI 不同

**解决：**
```python
# 错误
json={"model": "embo-01", "input": [text]}
result["data"][0]["embedding"]

# 正确
json={"model": "embo-01", "texts": [text], "type": "query"}
result["vectors"][0]
```

#### 3. MiniMax Embedding 余额不足
**问题：** `ValueError: MiniMax Embedding 错误: {'status_code': 1008, 'status_msg': 'insufficient balance'}`

**解决：** 改用本地免费 Embedding 模型 `BAAI/bge-small-zh-v1.5`

---

## 两个项目对比

| | 项目一 NLP2SQL | 项目二 RAG |
|---|---|---|
| 数据来源 | 结构化数据库（SQLite） | 非结构化文本（话术） |
| LLM 的角色 | 生成 SQL，不直接回答 | 基于检索结果生成回答 |
| 关键技术 | Function Calling | Embedding + 向量检索 |
| 适用场景 | 查数据、出报表 | 客服、知识问答 |

两个项目都在解决同一个问题：**让 LLM 用上真实数据，而不是靠"幻觉"回答。** 只是数据形态不同，用了不同的技术路线。

