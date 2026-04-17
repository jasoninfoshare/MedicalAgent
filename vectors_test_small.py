"""
vectors_test_small.py - 小批量测试（2000条），适合无显卡或低配机器快速验证
医疗健康 AI Agent 项目

运行方式：
    HF_HUB_OFFLINE=1 python vectors_test_small.py
    然后：HF_HUB_OFFLINE=1 python agent.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectors import MilvusVectorStore, prepare_documents

MILVUS_URI = "./milvus_agent.db"

if __name__ == "__main__":
    # 只加载 2000 条测试数据
    docs = prepare_documents(["./data/test_2000.jsonl"])
    if not docs:
        print("[ERROR] 没有加载到任何文档")
        exit(1)
    print(f"✅ 预处理完成，共 {len(docs)} 条文档（小批量测试）")

    store = MilvusVectorStore(uri=MILVUS_URI)
    store.create_vector_store(docs)
    print(f"✅ 向量库已保存到 {MILVUS_URI}，可以启动 agent.py 了")
