#!/usr/bin/env python3
"""最终搜索测试"""
import requests

# 测试几个典型的搜索查询
test_queries = [
    "连衣裙",
    "红色衣服",
    "T恤",
    "牛仔裤", 
    "鞋子",
    "时尚",
    "衣服"
]

print("测试搜索功能（数据库中有220张图像）:\n")

for query in test_queries:
    try:
        response = requests.post(
            'http://localhost:9000/search/text',
            data={'query': query, 'top_k': 3},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            filtered = result.get('filtered_results', 0)
            raw_sim = result.get('raw_similarities', [])
            
            if filtered > 0:
                print(f"'{query}': 找到 {filtered} 个结果")
                if raw_sim:
                    print(f"  相似度: {[f'{s:.3f}' for s in raw_sim[:3]]}")
            else:
                print(f"'{query}': 找到结果但相似度不足")
        else:
            print(f"'{query}': 错误 {response.status_code}")
    except Exception as e:
        print(f"'{query}': 服务未运行或连接失败")

print("\n提示: 确保服务正在运行 (python main.py)")

