#!/usr/bin/env python3
"""测试服务启动"""
import sys
import traceback

try:
    print("正在导入模块...")
    from src.search.semantic_search import SemanticSearchEngine
    
    print("正在初始化搜索引擎...")
    search_engine = SemanticSearchEngine()
    
    print("✓ 搜索引擎初始化成功！")
    
    print("\n正在获取系统信息...")
    info = search_engine.get_system_info()
    print(f"数据库: {info['database']['collection_name']}")
    print(f"图像数量: {info['database']['total_images']}")
    print(f"模型: {info['model']['name']}")
    print(f"设备: {info['model']['device']}")
    
    print("\n✓ 所有组件运行正常！")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    traceback.print_exc()
    sys.exit(1)

