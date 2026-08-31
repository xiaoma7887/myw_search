#!/usr/bin/env python3
"""直接检查数据库"""
from src.search.semantic_search import SemanticSearchEngine

engine = SemanticSearchEngine()
info = engine.get_system_info()
print(f"数据库图像数量: {info['database']['total_images']}")

