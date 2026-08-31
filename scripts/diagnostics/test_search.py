#!/usr/bin/env python3
"""测试搜索功能"""
import requests

# 测试文本搜索
print("测试文本搜索...")
queries = ["红色连衣裙", "运动鞋", "牛仔裤", "白色T恤"]

for query in queries:
    print(f"\n搜索: {query}")
    try:
        response = requests.post(
            'http://localhost:9000/search/text',
            data={'query': query, 'top_k': 5}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  找到 {result.get('total_results', 0)} 个结果")
            if result.get('results', {}).get('image_paths'):
                for i, path in enumerate(result['results']['image_paths'][:3], 1):
                    similarity = result['results']['similarities'][i-1]
                    print(f"  {i}. {path} (相似度: {similarity:.3f})")
            else:
                print("  没有找到匹配的结果")
        else:
            print(f"  错误: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  异常: {e}")

print("\n测试完成！")

