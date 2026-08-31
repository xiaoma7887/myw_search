#!/usr/bin/env python3
"""测试真实搜索功能"""
import requests
import json

# 测试不同的搜索查询
test_queries = [
    "连衣裙",
    "红色衣服", 
    "T恤",
    "牛仔裤",
    "鞋子",
    "运动鞋",
    "短袖",
    "裙子",
    "外套",
    "裤子",
    "白色",
    "黑色",
    "时尚",
    "服装"
]

print("测试文本搜索功能...\n")
successful_queries = []

for query in test_queries:
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
            
            if filtered > 0:
                print(f"✓ '{query}': 找到 {filtered} 个结果")
                successful_queries.append(query)
                # 显示前3个结果
                if result.get('results', {}).get('image_paths'):
                    for i, path in enumerate(result['results']['image_paths'][:3], 1):
                        sim = result['results']['similarities'][i-1]
                        print(f"    {i}. {path} (相似度: {sim:.3f})")
            else:
                print(f"  '{query}': 找到 {total} 个结果但相似度不足")
        else:
            print(f"✗ '{query}': 错误 {response.status_code}")
    except Exception as e:
        print(f"✗ '{query}': 异常 - {e}")

print(f"\n总结: 成功搜索到结果的查询: {successful_queries}")
print(f"\n建议: 可以尝试搜索这些关键词: {', '.join(successful_queries) if successful_queries else '暂无结果'}")

