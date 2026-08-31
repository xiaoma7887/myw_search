#!/usr/bin/env python3
"""
数据获取总控脚本
提供多种无需认证的数据获取方案
"""

import os
import sys

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_menu():
    """显示菜单"""
    print("=" * 60)
    print(" 电商图文数据集获取工具")
    print("=" * 60)
    print("\n请选择数据获取方案:")
    print("1.  Kaggle数据集 (推荐)")
    print("2.  学术公开数据集")
    print("3.  网络公开数据")
    print("4.  手动下载说明")
    print("5.  退出")
    print("\n" + "=" * 60)

def option_kaggle():
    """Kaggle数据集选项"""
    print("\n Kaggle数据集方案")
    print("-" * 30)
    
    try:
        from data_collection.kaggle_downloader import KaggleDatasetDownloader
        
        downloader = KaggleDatasetDownloader()
        
        # 显示推荐数据集
        datasets = downloader.get_recommended_datasets()
        
        print("\n推荐的数据集:")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset['name']}")
            print(f"   描述: {dataset['description']}")
            print(f"   样本: {dataset['samples']}")
        
        choice = input("\n选择要下载的数据集编号 (1-4): ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(datasets):
            dataset_index = int(choice) - 1
            dataset_name = datasets[dataset_index]['name']
            
            print(f"\n开始下载: {dataset_name}")
            
            success = downloader.download_dataset(
                dataset_name,
                output_dir=f"data/raw/kaggle/dataset_{choice}"
            )
            
            if success:
                print(" 下载完成!")
                print("下一步: 运行 python scripts/data_collection/dataset_builder.py")
            else:
                print(" 下载失败，请检查Kaggle配置")
        else:
            print(" 无效选择")
            
    except Exception as e:
        print(f" 执行失败: {e}")
        print("请确保已安装kaggle包: pip install kaggle")

def option_academic():
    """学术数据集选项"""
    print("\n 学术公开数据集")
    print("-" * 30)
    
    academic_datasets = [
        {
            'name': 'Fashion-MNIST',
            'url': 'https://github.com/zalandoresearch/fashion-mnist',
            'description': '7万张时尚商品图像，10个类别',
            'size': '~30MB',
            'format': '图像 + 标签'
        },
        {
            'name': 'DeepFashion',
            'url': 'http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html',
            'description': '80万张时尚图像，丰富属性标注',
            'size': '~5GB',
            'format': '图像 + 属性'
        },
        {
            'name': 'Amazon Product Data',
            'url': 'http://jmcauley.ucsd.edu/data/amazon/',
            'description': '亚马逊商品评论和元数据',
            'size': '~10GB',
            'format': 'JSON评论数据'
        }
    ]
    
    print("\n推荐的学术数据集:")
    for i, dataset in enumerate(academic_datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   描述: {dataset['description']}")
        print(f"   大小: {dataset['size']}")
        print(f"   链接: {dataset['url']}")
        print()
    
    print(" 操作说明:")
    print("1. 访问上述链接手动下载")
    print("2. 将数据解压到 data/raw/academic/ 目录")
    print("3. 运行 python scripts/data_collection/dataset_builder.py")

def option_web():
    """网络数据选项"""
    print("\n 网络公开数据")
    print("-" * 30)
    
    try:
        from data_collection.web_crawler import WebDataCollector
        
        collector = WebDataCollector()
        
        print("正在搜索公开数据源...")
        datasets = collector.collect_from_public_api()
        
        if datasets:
            print("\n找到的数据源:")
            for dataset in datasets:
                print(f"- {dataset['name']}: {dataset['description']}")
        else:
            print("未找到可用的公开数据源")
            
    except Exception as e:
        print(f"执行失败: {e}")

def option_manual():
    """手动下载说明"""
    print("\n 手动下载说明")
    print("-" * 30)
    
    instructions = """
 手动获取数据的步骤:

1. **寻找数据源**
   - 搜索 "ecommerce dataset"
   - 访问 data.world, data.gov 等开放数据平台
   - 查找学术论文的配套数据集

2. **下载数据**
   - 将数据下载到 data/raw/manual/ 目录
   - 支持格式: ZIP, JSON, CSV, 图像文件夹

3. **整理数据**
   - 确保每个商品有图像和对应文本描述
   - 图像格式: JPG, PNG, WebP
   - 文本格式: 商品标题、描述、属性等

4. **构建数据集**
   - 运行: python scripts/data_collection/dataset_builder.py
   - 脚本会自动处理数据格式和分割

 推荐的目录结构:

data/raw/manual/
├── images/           # 商品图像
├── metadata.json     # 商品元数据
└── descriptions.txt  # 商品描述

 提示: 可以从多个来源组合数据，构建更丰富的数据集
"""
    
    print(instructions)

def main():
    """主函数"""
    while True:
        show_menu()
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == '1':
            option_kaggle()
        elif choice == '2':
            option_academic()
        elif choice == '3':
            option_web()
        elif choice == '4':
            option_manual()
        elif choice == '5':
            print("\n 再见!")
            break
        else:
            print(" 无效选项，请重新选择")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()