#!/usr/bin/env python3
"""检查距离和相似度"""
import requests
from src.search.semantic_search import SemanticSearchEngine

# 直接使用搜索引擎，不通过API
engine = SemanticSearchEngine()

query = "连衣裙"
print(f"搜索: '{query}'")

results = engine.search_by_text(query, top_k=5)

print(f"\n总结果数: {results['total_results']}")
print(f"过滤后结果数: {results['filtered_results']}")

# 查看原始搜索结果中的distances
# 需要修改代码来查看原始distances，或者直接查看数据库搜索返回的结果

# 让我们直接测试数据库搜索
from src.database.chroma_db import ChromaDBManager
import yaml

with open("config/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

db_manager = ChromaDBManager("config/config.yaml")
text_emb = engine.clip_model.get_text_embedding(query)
raw_results = db_manager.search_by_text(text_emb, 5)

print(f"\n原始搜索结果:")
print(f"distances: {raw_results['distances']}")
if raw_results['distances']:
    similarities = [1 - d for d in raw_results['distances']]
    print(f"相似度: {similarities}")
    print(f"最高相似度: {max(similarities):.3f}")
    print(f"最低相似度: {min(similarities):.3f}")
    print(f"平均相似度: {sum(similarities)/len(similarities):.3f}")

