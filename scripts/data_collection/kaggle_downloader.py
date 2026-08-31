#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kaggle数据集下载工具
用于获取公开的电商图文数据集
"""

import sys
import io

# 设置Unicode支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # 对于旧版本Python
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import zipfile
import kaggle
import json
from typing import List, Dict, Any


class KaggleDatasetDownloader:
    """Kaggle数据集下载器"""
    
    def __init__(self):
        """初始化下载器"""
        # 检查kaggle配置
        self._check_kaggle_config()
    
    def _check_kaggle_config(self):
        """检查Kaggle配置"""
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json):
            print("⚠️  未找到Kaggle API配置")
            print("请按照以下步骤配置:")
            print("1. 访问 https://www.kaggle.com/")
            print("2. 注册账号并登录")
            print("3. 点击头像 → Account")
            print("4. 找到API区域，点击'Create New API Token'")
            print("5. 下载kaggle.json文件")
            print("6. 将文件移动到 ~/.kaggle/ 目录")
            return False
        return True
    
    def download_dataset(self, dataset_name: str, output_dir: str = "data/raw/kaggle") -> bool:
        """
        下载Kaggle数据集
        
        Args:
            dataset_name: 数据集名称
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📥 正在下载数据集: {dataset_name}")
            
            # 使用kaggle API下载
            kaggle.api.dataset_download_files(
                dataset_name,
                path=output_dir,
                unzip=True
            )
            
            print(f"✅ 数据集下载完成: {output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def get_recommended_datasets(self) -> List[Dict[str, str]]:
        """
        获取推荐的电商数据集列表
        """
        return [
            {
                'name': 'paramaggarwal/fashion-product-images-dataset',
                'description': '时尚商品图像数据集',
                'size': '~1.2GB',
                'samples': '44,000+ 图像',
                'format': '图像 + CSV元数据'
            },
            {
                'name': 'saurabhshahane/ecommerce-text-classification',
                'description': '电商文本分类数据集',
                'size': '~50MB',
                'samples': '50,000+ 商品描述',
                'format': 'CSV文本数据'
            },
            {
                'name': 'prajjwal1/clothing-fit-dataset',
                'description': '服装合身度数据集',
                'size': '~200MB',
                'samples': '20,000+ 评论和图像',
                'format': '图像 + 文本'
            },
            {
                'name': 'validmodel/amazon-product-reviews-2023',
                'description': '亚马逊商品评论数据集',
                'size': '~800MB',
                'samples': '100,000+ 评论',
                'format': 'JSON评论数据'
            }
        ]


def main():
    """主函数"""
    downloader = KaggleDatasetDownloader()
    
    print("🛒 推荐的电商图文数据集:")
    datasets = downloader.get_recommended_datasets()
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{i}. {dataset['name']}")
        print(f"   描述: {dataset['description']}")
        print(f"   大小: {dataset['size']}")
        print(f"   样本: {dataset['samples']}")
        print(f"   格式: {dataset['format']}")
    
    print("\n🚀 开始下载第一个推荐数据集...")
    
    if datasets:
        # 下载第一个数据集
        success = downloader.download_dataset(
            datasets[0]['name'],
            output_dir="data/raw/kaggle/fashion"
        )
        
        if success:
            print("\n✅ 数据集下载完成!")
            print("下一步: 使用 dataset_builder.py 处理数据")


if __name__ == "__main__":
    main()