"""
model.py - 模型搭建与引用
医疗健康 AI Agent 项目

依赖安装：
    pip install openai langchain-huggingface sentence-transformers
"""

import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

# ============================================================
# 配置区：填入你自己的 API Key
# ============================================================
MINIMAX_API_KEY  = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL    = "MiniMax-M2.5"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")

# 本地 bce embedding 模型（首次运行自动从 HuggingFace 下载到缓存）
BCE_EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"


# ============================================================
# 模型1：本地 HuggingFace Embedding（bce-embedding-base_v1）
# ============================================================
def create_embedding_model(model_name: str = BCE_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    加载本地 bce-embedding-base_v1 嵌入模型。
    首次运行会自动下载（约 500 MB），之后从本地缓存加载。
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "batch_size": 128,
            "normalize_embeddings": True,  # 归一化，适配内积距离 IP
        },
        model_kwargs={"local_files_only": True},
    )
    return embeddings


# ============================================================
# 模型2：智谱 ZhipuAI Embedding（embedding-3）
# ============================================================
class ZhipuAIEmbeddings(Embeddings):
    """智谱 AI Embedding-3 封装（需安装 zhipuai）"""
    def __init__(self, client=None, api_key: str = None):
        if client is not None:
            self.client = client
        else:
            try:
                from zhipuai import ZhipuAI
                self.client = ZhipuAI(api_key=api_key or ZHIPU_API_KEY)
            except ImportError:
                raise ImportError("请安装 zhipuai: pip install zhipuai")

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model="embedding-3",
                input=[text],
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def create_zhipu_embedding_model() -> ZhipuAIEmbeddings:
    """创建智谱 Embedding-3 模型"""
    return ZhipuAIEmbeddings()


# ============================================================
# 模型3：MiniMax API（OpenAI 兼容接口）
# ============================================================
def create_minimax_client() -> OpenAI:
    """创建 MiniMax OpenAI 兼容客户端"""
    client = OpenAI(
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
    )
    return client


def generate_minimax_answer(client: OpenAI, prompt: str) -> str:
    """调用 MiniMax-M2.5 生成回答"""
    response = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=[
            {"role": "system", "content": "你是一个能力非常强大的医学助手。"},
            {"role": "user",   "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content


# ============================================================
# 模型4：DeepSeek API（OpenAI 兼容接口）
# ============================================================
def create_deepseek_client() -> OpenAI:
    """创建 DeepSeek 客户端"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return client


def generate_deepseek_answer(client: OpenAI, prompt: str) -> str:
    """调用 DeepSeek 生成回答"""
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是一个能力非常强大的助手。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content


# ============================================================
# 模型5：本地 Qwen3-Next 模型（可选，需要 GPU）
# ============================================================
def create_local_qwen_model(model_path: str = "./Qwen3-Next-80B-A3B-Thinking"):
    """加载本地 Qwen3 模型（需要 transformers + torch）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return model, tokenizer


def generate_local_qwen_answer(model, tokenizer, question: str) -> str:
    """使用本地 Qwen3 生成回答"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content.strip()


# ============================================================
# 快速测试入口
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("测试 bce-embedding（首次运行需下载模型约 500MB）...")
    emb = create_embedding_model()
    vec = emb.embed_query("糖尿病患者的饮食建议")
    print(f"向量维度：{len(vec)}")
    print("模型测试通过 ✅")

    if DEEPSEEK_API_KEY:
        print("=" * 50)
        print("测试 DeepSeek API...")
        client = create_deepseek_client()
        answer = generate_deepseek_answer(client, "你好，请简单介绍一下你自己")
        print(answer)
    elif MINIMAX_API_KEY:
        print("=" * 50)
        print("测试 MiniMax API...")
        client = create_minimax_client()
        answer = generate_minimax_answer(client, "你好，请简单介绍一下你自己")
        print(answer)
    else:
        print("⚠️ 未配置 DEEPSEEK_API_KEY 或 MINIMAX_API_KEY，跳过 LLM 测试")