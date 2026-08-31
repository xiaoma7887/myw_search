#!/usr/bin/env python3
"""
快速设置数据库 - 添加少量图像用于测试
"""
import os
import glob
from src.search.semantic_search import SemanticSearchEngine

def quick_setup(num_images=100):
    """快速添加指定数量的图像"""
    print("初始化搜索引擎...")
    search_engine = SemanticSearchEngine()
    
    # 获取初始状态
    info = search_engine.get_system_info()
    print(f"当前数据库图像数量: {info['database']['total_images']}")
    
    # 查找图像文件
    image_dir = "data/raw/kaggle/fashion/fashion-dataset/images"
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))[:num_images]
    
    print(f"\n找到 {len(image_paths)} 张图像，开始添加...")
    
    # 分批添加
    batch_size = 20
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} ({len(batch)} 张)...")
        
        result = search_engine.add_images_to_index(batch)
        if result['success']:
            print(f"  成功添加 {result['added_count']} 张")
        else:
            print(f"  失败: {result.get('message', '未知错误')}")
    
    # 最终状态
    final_info = search_engine.get_system_info()
    print(f"\n完成！数据库中共有 {final_info['database']['total_images']} 张图像")
    print("\n现在可以开始搜索了！")
    print("示例搜索:")
    print("  - 文本搜索: POST http://localhost:9000/search/text")
    print("  - 图像搜索: POST http://localhost:9000/search/image")

if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    quick_setup(num)

