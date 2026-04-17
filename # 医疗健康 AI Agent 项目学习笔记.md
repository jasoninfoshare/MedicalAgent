# 医疗健康 AI Agent 项目学习笔记

---

## 一、项目整体架构

### 技术栈
| 组件 | 选型 |
|---|---|
| LLM | MiniMax-M2.5 |
| Embedding | bce-embedding-base_v1（本地） |
| 向量数据库 | Milvus Lite |
| 关键词检索 | BM25Retriever |
| 混合融合 | EnsembleRetriever（RRF） |
| 服务框架 | FastAPI + Uvicorn |

### 整体流程
```
用户提问
   ↓
Agent.py（FastAPI 服务）
   ↓
① BM25 关键词检索 + ② Dense 语义检索 → EnsembleRetriever 融合
   ↓
从 Milvus Lite 向量库取出相关文档
   ↓
拼入 Prompt → MiniMax API 生成回答
   ↓
返回 JSON 给用户
```

---

## 二、项目文件说明

### 文件职责一览
| 文件 | 职责 |
|---|---|
| `model.py` | 两个模型的初始化工具箱 |
| `create_data.py` | 造知识库原材料 |
| `vectors.py` | 把原材料变成向量存入数据库 |
| `Agent.py` | 对外提供问答服务的 HTTP 接口 |

### model.py
```python
# 模型1：bce-embedding-base_v1
# 作用：把文字变成向量
def create_embedding_model():
    return HuggingFaceEmbeddings(model_name="bce-embedding-base_v1")

# 模型2：MiniMax-M2.5
# 作用：接收问题+上下文，生成最终回答
def create_minimax_client():
    return OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
```

### create_data.py
手写了 30 条医疗问答，保存成 jsonl 格式：
```json
{"query": "糖尿病患者平时应该注意哪些饮食？", "response": "...详细回答..."}
{"query": "高血压患者日常生活中需要注意什么？", "response": "..."}
```

### vectors.py
```
运行前：只有 data.jsonl
运行后：多了 milvus_agent.db（向量数据库）
```
流程：读取 jsonl → bce-embedding 把文字变向量 → 存入 Milvus Lite

### Agent.py
**启动时（只执行一次）：**
```python
embedding_model = create_embedding_model()     # 加载本地模型
milvus_vectorstore = Milvus(...)               # 连接向量库
bm25_retriever = BM25Retriever(...)            # 初始化 BM25
retriever = EnsembleRetriever(                 # 混合检索器
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)
client_llm = create_minimax_client()           # 连接 MiniMax
```

**每次提问时：**
```python
recalled_docs = retriever.invoke(query)        # 混合检索
context = format_docs(recalled_docs)           # 拼上下文
prompt = build_prompt(context, query)          # 构造 Prompt
response = generate_minimax_answer(client_llm, prompt)  # LLM生成
```

---

## 三、data.jsonl 和 train.jsonl 的区别

在本项目里**没有本质区别**，两个文件格式完全一样，代码里直接合并处理（共 30 条）。

这是沿用了机器学习的命名习惯：
- `data.jsonl` → 主知识库数据（20 条）
- `train.jsonl` → 补充知识库数据（10 条）

我们做的是 RAG，不需要训练模型，两个文件都只是"知识库原材料"。

---

## 四、向量数据库的好处

### 核心问题：快速找相似向量

没有向量数据库的暴力做法：和每一条文档逐一计算相似度，数据量大了极慢。

Milvus 用 IVF_FLAT 索引解决这个问题：
```
建库时：把向量先聚类，分成 128 个"桶"
查询时：先判断属于哪几个桶，只在桶里找
速度从 O(N) 降到接近 O(log N)
```

### 速度对比
| 数据量 | 暴力遍历 | Milvus |
|---|---|---|
| 1万条 | 0.5秒 | 5毫秒 |
| 100万条 | 50秒 | 10毫秒 |
| 1亿条 | 几乎不可用 | 50毫秒 |

### 和普通数据库的区别
```
MySQL：擅长精确匹配（SELECT WHERE name='张三'）
Milvus：擅长相似度匹配（找语义相近的内容）
```

用户问"血糖高怎么办"，知识库里存的是"糖尿病饮食建议"，MySQL 找不到，Milvus 能找到。

---

## 五、为什么要先加载模型再连接数据库

```python
embedding_model = create_embedding_model()  # 先加载模型
milvus_vectorstore = Milvus(
    embedding_function=embedding_model,      # 模型传入数据库
    connection_args={"uri": "./milvus_agent.db"}
)
```

Milvus 查询时需要把用户问题变成向量，必须借用 embedding 模型。
模型是工具，数据库是仓库，**仓库需要工具才能工作，所以工具必须先准备好。**

---

## 六、本地 embedding 模型 vs GLM embedding API

| 维度 | 本地 bce-embedding | 智谱 embedding-3 |
|---|---|---|
| 速度 | 快（本地计算） | 慢（网络请求） |
| 费用 | 免费 | 按调用次数收费 |
| 隐私 | 数据不出本机 | 数据发送到服务器 |
| 稳定性 | 离线可用 | 依赖网络 |
| 向量维度 | 768维 | 2048维 |
| 存储 | 本地 1.1GB | 不占本地空间 |

**选本地模型的原因：** 免费、快、医疗数据不外泄、离线可用。

---

## 七、embedding 模型和 Milvus 能用 GPU 吗

### embedding 模型：可以
```python
embeddings = HuggingFaceEmbeddings(
    model_name=BCE_EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},   # 加这一行
    encode_kwargs={"batch_size": 64},  # GPU 可以加大批量
)
```

检查 GPU 是否可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

GPU 加速效果：
```
CPU 处理 100 条：5-8 秒
GPU 处理 100 条：0.5-1 秒
```

### Milvus Lite：不支持 GPU
Milvus Lite 是纯 CPU 轻量版，完整版 Milvus（Docker 部署）支持 GPU。
我们这个规模（30条数据）CPU 已经足够。

---

## 八、换成 GLM embedding 还需要 Milvus 吗

**需要**，两者职责完全不同：
```
embedding 模型：负责"把文字变成向量"
Milvus：负责"存储和检索向量"
```

类比：
```
embedding 模型 = 翻译官（把中文翻译成数字）
Milvus        = 图书馆（存数字、找数字）
换翻译官不影响图书馆的存在
```

---

## 九、BM25、Dense、RRF、父子文档的区别

### 四者解决不同层面的问题
```
BM25、Dense      → 解决"用什么方式搜索"
RRF 混合检索     → 解决"多路结果怎么合并排序"
父子文档检索     → 解决"找到后返回多少内容"
```

### BM25 原理（词频统计）
```
用户问："糖尿病 饮食"
→ 统计哪些文档包含这两个词
→ 按词频打分排序
→ 擅长精确词匹配，不理解语义
```

### Dense 原理（向量距离）
```
用户问："血糖高怎么办"
→ 变成向量 [0.12, -0.34, 0.56, ...]
→ 计算余弦相似度，找最近的向量
→ 擅长语义理解，能关联"血糖高"和"糖尿病"
```

### RRF 融合原理
```python
# BM25结果：[文档A(第1), 文档C(第2), 文档E(第3)]
# Dense结果：[文档A(第1), 文档B(第2), 文档C(第3)]

# RRF公式：得分 = Σ 1/(k + 排名)，k=60
文档A：1/(60+1) + 1/(60+1) = 0.0328  # 两路都第一，得分最高
文档C：1/(60+2) + 1/(60+3) = 0.0318  # 两路都靠前
文档B：0        + 1/(60+2) = 0.0161  # 只有Dense召回

# 核心思想：多路检索都靠前的文档才是真正相关的
```

### 父子文档检索原理
```
建库：小块（200字）建索引，大块（1000字）保留原文
查询：命中小块 → 返回对应大块
优点：检索精准 + 上下文完整
```

### 四者串联关系
```
用户问题
   ↓
BM25 检索 → 5篇文档
Dense 检索 → 5篇文档
   ↓
RRF 融合 → 合并成1个排好序的列表
   ↓
父子文档 → 命中小块，返回大块
   ↓
送给 LLM
```

---

## 十、Markdown 格式的好处及 PDF 转 Markdown

### Markdown 对 RAG 的三大优势

**① 结构保留**
```
pdfplumber 提取：乱成一团的纯文本
MarkItDown 转换：# 标题、## 小节、- 列表，结构清晰
```

**② 分块更准确**
```python
# 按标题分块，每块都是完整知识单元
MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "章"), ("##", "节")]
)
```

**③ LLM 理解更好**
LLM 训练数据包含大量 Markdown，结构化输入回答质量更高。

### PDF 转 Markdown 方案
```bash
pip install markitdown
```
```python
from markitdown import MarkItDown
md = MarkItDown()
result = md.convert("内科学_第9版.pdf")
markdown_text = result.text_content
```

### MarkItDown 原理
```
PDF 文件
   ↓
解析 PDF 内部绘图指令
   ↓
字体大小 > 16px → 判断为标题 → 转成 #
字体加粗 → 转成 **文字**
检测横线竖线 → 识别表格 → 转成 Markdown 表格
检测列表符号 → 转成 - 格式
   ↓
输出 Markdown
```

扫描版 PDF 额外走 OCR 流程（Tesseract 或 Azure Vision）。

---

## 十一、工业界常用技术栈

### 完整 RAG 流程各层技术

**数据处理层**
```
pdfplumber / PyMuPDF / Unstructured / MarkItDown
```

**向量数据库**
```
Milvus（十亿级）/ Qdrant（新项目首选）/ Weaviate / Pinecone / ES
```

**索引算法**
```
IVF_FLAT（我们用）/ HNSW（工业界最常用）/ ANNOY
```

**检索层**
```
Dense + BM25 混合检索（标配）
知识图谱检索（关系复杂场景）
查询改写 / RAG-Fusion / Step-back / HyDE
```

**后处理层**
```
Reranker 重排序（bge-reranker / bce-reranker / qwen3-reranker）
上下文压缩（LLMChainExtractor）
MMR 去重
```

**生成层**
```
流式输出（stream=True）
严格 Prompt 约束
RAG + Agent 工具调用
```

**评估层**
```
RAGAS（忠实度、答案相关性、上下文精确率、召回率）
TruLens
Bad Case 人工分析
```

### 我们项目 vs 工业界
```
已实现：
  ✅ BM25 + Dense 混合检索
  ✅ Milvus 向量库
  ✅ 父子文档检索
  ✅ RRF 融合排序
  ✅ Reranker 重排序

待完善：
  ⬜ 流式输出
  ⬜ 查询改写
  ⬜ 知识图谱
  ⬜ RAGAS 评估
  ⬜ 前端页面
```

覆盖工业界 RAG 系统约 **70%** 的核心技术。

# 医疗健康 AI Agent 项目开发 — 对话学习笔记

> 整理自与 Claude 的学习对话，涵盖 Agent开发4、5、6 三份讲义内容。

---

## 📄 Agent开发4 — 项目工具安装 + neo4j优化RAG + 缓存命中

### 第1章：项目开发工具安装

#### 1.1 Redis 安装（macOS 源码编译）

1. 官网下载 Redis 7.0.15 稳定版：https://redis.io/
2. 解压后移动到 `/usr/local`
3. 进入目录编译：`sudo make test`（约5分钟）
4. 安装：`sudo make install`
5. 在 redis 目录下新建 `bin`、`etc`、`db` 三个文件夹
6. 将 `src` 下的5个文件拷贝到 `bin` 目录
7. 修改 `redis.conf` 配置：
   ```conf
   daemonize yes        # 后台启动
   protected-mode no    # 关闭保护模式
   appendonly yes       # 打开AOF功能
   ```
8. 启动服务：`redis-server redis.conf`
9. 启动命令行：`redis-cli`
10. 可视化工具：https://redis.io/insight

#### 1.2 Milvus 向量数据库安装

```bash
pip install pymilvus
pip install pymilvus-model
```

---

### 第2章：知识图谱 neo4j 优化 RAG

#### 核心问题

工业界痛点：**LLM 不可靠性** vs **数据库查询正确性**，不允许错误查询命令执行。

#### 解决思路（4个约束）

1. 严格定义数据 schema，让 LLM 只在有限集合中选择
2. 严格检查 LLM 生成的查询命令是否满足规范
3. 严格用解释器对命令进行验证
4. 对最终返回的执行命令格式进行验证

#### 架构设计（5个模块）

| 文件 | 职责 |
|------|------|
| `schemas.py` | 图数据库 schema 定义（节点、关系） |
| `prompts.py` | 提示词功能定义（含 Few-shot 示例） |
| `models.py` | 请求类与响应类的字段定义（Pydantic） |
| `validators.py` | 验证类（语法验证 + schema 验证 + 规则验证） |
| `main.py` | FastAPI 服务，端口 8101 |

#### 医疗知识图谱 Schema 示例

```python
nodes = [Disease, Drug, Food, Symptom]
relationships = [
    Disease --has_symptom--> Symptom,
    Disease --recommand_drug--> Drug,
    Disease --recommand_eat--> Food,
]
```

#### 验证器设计

- `CypherValidator`：连接真实 neo4j，用 `EXPLAIN` 命令验证语法
- `RuleBasedValidator`：无法连接 neo4j 时的备用规则验证器（拦截 DROP/DELETE/DETACH 等危险操作）

---

### 第3章：缓存命中（v1.2）

#### 整体请求流程

```
请求 → Redis缓存查询 → 命中则直接返回
              ↓ 未命中
       Milvus 模糊召回（top-10，RRF重排）
              ↓
       neo4j 精准召回（Cypher查询）
              ↓
       DeepSeek 生成答案
              ↓
       写入 Redis 缓存
```

#### 关键代码（vectors.py）

```python
def cache_set(r, question: str, answer: str):
    r.hset("qa", question, answer)
    r.expire("qa", 3600)

def cache_get(r, question: str):
    return r.hget("qa", question)
```

#### v1.2 存在的生产隐患

1. **大 Key 问题**：所有问答对存入同一个 Hash，百万级数据时导致 Redis 阻塞
2. **连接池管理**：每次调用都创建新连接池，高并发时耗尽 TCP 资源
3. **缺乏并发保护**：没有防止缓存击穿的机制
4. **缺乏雪崩保护**：过期时间固定，大量缓存同时失效

---

## 📄 Agent开发5 — Redis v2.0 优化

### 四个问题的解决方案

#### 问题1：大 Key → 独立 String Key

```python
def _generate_key(self, text: str, prefix: str = "llm:cache:") -> str:
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return f"{prefix}{hash_obj.hexdigest()}"
    # 示例: "llm:cache:e10adc3949ba59abbe56e057f20f883e"
```

#### 问题2：连接池管理 → 单例模式

```python
class RedisClientWrapper:
    _pool = None  # 类变量，全局共享

    def __init__(self, ...):
        if not RedisClientWrapper._pool:
            RedisClientWrapper._pool = redis.ConnectionPool(
                max_connections=100, ...
            )
        self.client = redis.StrictRedis(connection_pool=RedisClientWrapper._pool)

redis_manager = RedisClientWrapper()  # 全局只实例化一次
```

> **比喻**：v1.2 每次喝水都打一口新井；v2.0 全村共用一口井，借桶用完还回去。

#### 问题3：缓存击穿 → 分布式互斥锁 + Double Check

```python
def get_or_compute(self, question, compute_func):
    # 1. 查缓存
    cached = self.get_answer(question)
    if cached: return cached

    # 2. 加锁
    lock_token = self.acquire_lock(hash_key)
    if lock_token:
        try:
            # 3. Double Check（关键！防止重复调用LLM）
            cached_retry = self.get_answer(question)
            if cached_retry: return cached_retry

            # 4. 调用 LLM
            answer = compute_func()
            self.set_answer(question, answer)
            return answer
        finally:
            self.release_lock(hash_key, lock_token)
```

**为什么需要 Double Check？**
- 100个并发请求同时 Cache Miss，只有线程A抢到锁去调 LLM
- 线程B抢到锁后，若无 Double Check 会再次调用 LLM（浪费100倍费用）
- 有 Double Check：线程B发现缓存已有数据，直接返回，只调用1次 LLM

**为什么用 UUID 作为锁的值？**
- 防止锁误删：线程A超时后不能删除线程B持有的锁
- UUID 区分的是**持有锁的线程实例**，不是业务用户
- 通过 Lua 脚本原子性释放（比较 UUID 后再 DEL）

#### 问题4：缓存雪崩 → 随机过期时间抖动

```python
def set_answer(self, question, answer, expire_time=3600):
    jitter = random.randint(int(-expire_time * 0.1), int(expire_time * 0.1))
    real_expire = expire_time + jitter  # ±10% 随机抖动
    self.client.setex(key, real_expire, answer)
```

#### 防缓存穿透：`<EMPTY>` 占位符

当 LLM 对某问题返回空时，写入 `<EMPTY>` 占位符：
```python
self.client.setex(key, 60, "<EMPTY>")
```
后续相同请求命中 `<EMPTY>` 直接拦截，不再调用 LLM。

### v2.0 效果对比

| 版本 | 首次查询 | 缓存命中 |
|------|---------|---------|
| v1.2 | ~12s | ~10ms |
| v2.0 | ~10s | **<5ms** |

---

## 📄 Agent开发6 — RAG评估体系 + 企业需求案例

### 面试题：RAG的评估体系怎么做？

> 核心观点：RAG 评估不只是算一个分数，而是判断系统是不是**稳**、是不是**准**、是不是**能上线**。

#### 三层评估框架

##### 第一层：检索评估（Retrieval Evaluation）

| 指标 | 说明 |
|------|------|
| **Recall@K** | 检索结果中包含正确文档的比例，代表覆盖面 |
| **Precision@K** | 检索结果中真正相关的比例 |
| **MRR** | 第一个正确答案出现的排名 |
| **nDCG** | 加权排序指标，越靠前的相关文档权重越高 |

##### 第二层：生成一致性评估（Generation Consistency）

**自动化指标：**
- **Faithfulness Score**：生成答案与检索材料的语义相似度
- **Groundedness Score**：每条结论是否能在检索片段中找到依据
- **Factual Consistency**：基于 LLM 自检，判断答案是否自洽、是否编造

**人工评测指标：**
- 正确性（Correctness）
- 完整性（Completeness）
- 可读性（Fluency）
- 引用充分性（Attribution）

##### 第三层：系统性评估（System Evaluation）

| 指标 | 说明 |
|------|------|
| 延迟（Latency） | 检索+生成全流程响应时间 |
| 吞吐（Throughput） | 高并发下性能稳定性 |
| 缓存命中率 | 是否重复计算 |
| 可复现性 | 同样问题是否输出一致 |
| 时效性（Recency） | 知识库更新后是否实时反映 |
| Error Rate | 生成失败或超时的比例 |
| Rejection Rate | 模型拒答率（说明数据覆盖不足） |

#### 评估实施五步骤

1. **离线验证（Offline Eval）**：标注集跑 Recall@K、Faithfulness、BLEU、ROUGE
2. **灰度验证（A/B Test）**：在部分真实流量上对比新旧版本效果
3. **在线监控（Online Eval）**：持续监控 Error Rate、Rejection Rate、Drift Monitor、用户点赞率
4. **高风险行业评估**：金融/医疗强调可追溯、引用权威性（检索层 + 生成层 + 审查层三重验证）
5. **用户体验**：用户满意度 + 答案可读性

#### 代码示例（BLEU + ROUGE + 覆盖率）

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

ref = "您的汽车保险可赔偿医疗费用、车辆维修费，以及第三方损害赔偿。"
gen = "您的保单通常涵盖车祸后的医疗费用、车辆损失，以及对第三方的赔偿。"

bleu = sentence_bleu([list(ref)], list(gen), smoothing_function=SmoothingFunction().method1)
rouge = Rouge().get_scores(gen, ref)
print(f"BLEU: {bleu:.3f}, ROUGE-1 F1: {rouge[0]['rouge-1']['f']:.3f}")

# 支持文档覆盖率
tokens = lambda x: [c for c in x if c.strip()]
a, d = set(tokens(generated)), set(tokens("".join(docs)))
coverage = len(a & d) / len(a)
print(f"支持文档覆盖率: {coverage:.2f}")
```

#### 面试标准答案模板

> "我们制定了一个五维度的评估体系，从检索评估、生成一致性评估、系统性评估，一些列指标全面衡量系统表现。检索评估通过 BLEU、ROUGE、MRR、nDCG、Top-k 召回率衡量；生成一致性评估包含了自动化评估与人工评测；系统性评估考察了延迟、吞吐量、缓存命中、可复现、时效性等指标..."

**关键**：这样回答能让面试官感觉你不只是在"做RAG项目"，而是在**做RAG产品**。

---

### 现实企业级 AI 需求案例（9个）

| # | 场景 | 核心功能 |
|---|------|---------|
| 1 | AI培训考核 | 语音收集讲话 → AI出题判分 → 自动通过/补考 |
| 2 | 母婴AI销售机器人 | 扫码领红包 → AI语音引导推销母婴产品 |
| 3 | AI旅游导游健康监测 | 监测导游用嗓强度和身体姿态，生成健康日报 |
| 4 | AI虚拟女友 | 好感度系统 + 性格难度档位 + Stripe付费订阅 |
| 5 | AI智慧屏幕 | 能听/会说/能看/懂思考的数字人，支持实时字幕 |
| 6 | 养老陪伴机器人 | 语音唤醒提醒用药，带热敏打印机和SOS按钮 |
| 7 | 政务AI数字人 | 本地化部署，支持知识库上传，用于政府/医院/学校 |
| 8 | AI批改作业机器人 | 微信小程序拍照 + DeepSeek批改 + 学情分析 + 错题本 |
| 9 | 医药公司AI宣传 | 知识库 → 文生图/文生视频 → MCN分发 → 数据回流 |

---

*笔记整理时间：2026年4月*