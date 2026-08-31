#!/usr/bin/env python3
"""检查相似度分数"""
import requests
import json

queries = ["连衣裙", "红色衣服", "T恤", "鞋子", "时尚"]

print("检查搜索结果和相似度分数:\n")

for query in queries:
    try:
        response = requests.post(
            'http://localhost:9000/search/text',
            data={'query': query, 'top_k': 5},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            total = result.get('total_results', 0)
            filtered = result.get('filtered_results', 0)
            
            print(f"查询: '{query}'")
            print(f"  总结果数: {total}")
            print(f"  过滤后结果数: {filtered}")
            
            if result.get('results', {}).get('similarities'):
                similarities = result['results']['similarities']
                print(f"  相似度分数: {[f'{s:.3f}' for s in similarities[:5]]}")
                print(f"  最高相似度: {max(similarities) if similarities else 'N/A':.3f}")
                print(f"  最低相似度: {min(similarities) if similarities else 'N/A':.3f}")
            print()
    except Exception as e:
        print(f"错误 '{query}': {e}\n")

