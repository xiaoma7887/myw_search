#!/usr/bin/env python3
"""查看原始搜索结果"""
import requests
import json

query = "连衣裙"
response = requests.post(
    'http://localhost:9000/search/text',
    data={'query': query, 'top_k': 5},
    timeout=10
)

if response.status_code == 200:
    result = response.json()
    print("完整搜索结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"错误: {response.status_code}")
    print(response.text)

