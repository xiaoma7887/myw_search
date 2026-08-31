#!/usr/bin/env python3
"""
语义搜索系统使用示例
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search.semantic_search import SemanticSearchEngine


def example_usage():
    """使用示例"""
    
    # 初始化搜索引擎
    print("初始化语义搜索引擎...")
    search_engine = SemanticSearchEngine()
    
    # 获取系统信息
    print("\n=== 系统信息 ===")
    system_info = search_engine.get_system_info()
    print(f"数据库: {system_info['database']['collection_name']}")
    print(f"图像数量: {system_info['database']['total_images']}")
    print(f"模型: {system_info['model']['name']}")
    
    # 示例：添加图像到索引
    print("\n=== 添加图像到索引 ===")
    # 假设有一些图像文件
    example_images = [
        "data/raw/example1.jpg",
        "data/raw/example2.jpg",
        "data/raw/example3.jpg"
    ]
    
    # 只添加实际存在的图像
    existing_images = [img for img in example_images if os.path.exists(img)]
    
    if existing_images:
        result = search_engine.add_images_to_index(existing_images)
        print(f"添加结果: {result}")
    else:
        print("没有找到示例图像，跳过添加步骤")
    
    # 示例：文本搜索
    print("\n=== 文本搜索示例 ===")
    test_queries = [
        "红色连衣裙",
        "运动鞋",
        "笔记本电脑"
    ]
    
    for query in test_queries:
        print(f"\n搜索查询: '{query}'")
        results = search_engine.search_by_text(query, top_k=3)
        
        print(f"找到 {results['total_results']} 个结果")
        print(f"过滤后 {results['filtered_results']} 个结果")
        
        if results['results']['image_paths']:
            for i, (path, similarity) in enumerate(zip(
                results['results']['image_paths'], 
                results['results']['similarities']
            )):
                print(f"  {i+1}. {os.path.basename(path)} (相似度: {similarity:.3f})")
        else:
            print("  没有找到匹配的结果")
    
    # 示例：图像搜索
    print("\n=== 图像搜索示例 ===")
    if existing_images:
        query_image = existing_images[0]
        print(f"查询图像: {os.path.basename(query_image)}")
        
        results = search_engine.search_by_image(query_image, top_k=3)
        
        print(f"找到 {results['total_results']} 个结果")
        print(f"过滤后 {results['filtered_results']} 个结果")
        
        if results['results']['image_paths']:
            for i, (path, similarity) in enumerate(zip(
                results['results']['image_paths'], 
                results['results']['similarities']
            )):
                print(f"  {i+1}. {os.path.basename(path)} (相似度: {similarity:.3f})")
        else:
            print("  没有找到匹配的结果")


if __name__ == "__main__":
    example_usage()