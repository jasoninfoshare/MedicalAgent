"""
test_client.py - Agent 服务测试客户端
"""

import requests
import time
import json

url = "http://0.0.0.0:8103/"
data = {"question": "平日里蜂蜜加白醋一起喝有什么疗效?"}

start_time = time.time()
res = requests.post(url, json=data)
cost_time = time.time() - start_time

print('单次查询的耗时:', cost_time, 's')
print(res.json())
