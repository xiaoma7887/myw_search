#!/usr/bin/env python3
"""
数据库初始化脚本
用于批量导入图像到向量数据库
"""

import os
import argparse
import glob
from tqdm import tqdm
import yaml

from src.search.semantic_search import SemanticSearchEngine


def setup_database(image_dir: str, batch_size: int = 100):
    """
    初始化数据库，批量导入图像
    
    Args:
        image_dir: 图像目录路径
        batch_size: 批次大小
    """
    # 初始化搜索引擎
    search_engine = SemanticSearchEngine()
    
    # 获取系统信息
    system_info = search_engine.get_system_info()
    print(f"系统信息: {system_info}")
    
    # 查找所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for extension in image_extensions:
        pattern = os.path.join(image_dir, '**', extension)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    if not image_paths:
        print("未找到图像文件，请检查目录路径")
        return
    
    # 分批处理
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_paths)} 张图像)...")
        
        # 添加图像到索引
        result = search_engine.add_images_to_index(batch_paths)
        
        if result['success']:
            print(f"✓ 成功添加 {result['added_count']} 张图像")
        else:
            print(f"✗ 添加失败: {result['message']}")
    
    # 最终统计
    final_info = search_engine.get_system_info()
    print(f"\n数据库初始化完成！")
    print(f"总图像数量: {final_info['database']['total_images']}")


def main():
    parser = argparse.ArgumentParser(description='初始化向量数据库')
    parser.add_argument('--image_dir', type=str, required=True, 
                       help='包含图像的目录路径')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批次处理大小，默认100')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误: 目录不存在: {args.image_dir}")
        return
    
    setup_database(args.image_dir, args.batch_size)


if __name__ == "__main__":
    main()