#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替代数据集下载方案
提供更小、更可靠的数据集下载选项
"""

import sys
import io

# 设置Unicode支持
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import requests
import zipfile

def download_small_fashion_dataset():
    """下载较小的时尚数据集"""
    
    # Fashion-MNIST数据集（较小，约30MB）
    urls = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    ]
    
    output_dir = "data/raw/fashion_mnist"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在下载Fashion-MNIST数据集（较小版本）...")
    
    for url in urls:
        filename = os.path.basename(url)
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"下载: {filename}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"完成: {filename}")
            
        except Exception as e:
            print(f"下载失败 {filename}: {e}")
            return False
    
    print(f"\n数据集已保存到: {output_dir}")
    return True

def create_sample_dataset():
    """创建示例数据集用于测试"""
    
    output_dir = "data/raw/sample"
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # 创建示例数据
    sample_data = [
        {
            "image_path": "data/raw/sample/images/sample1.jpg",
            "text": "红色连衣裙 夏季新款",
            "category": "女装/连衣裙"
        },
        {
            "image_path": "data/raw/sample/images/sample2.jpg", 
            "text": "蓝色牛仔裤 修身款",
            "category": "女装/裤子"
        },
        {
            "image_path": "data/raw/sample/images/sample3.jpg",
            "text": "白色运动鞋 轻便舒适",
            "category": "鞋类/运动鞋"
        }
    ]
    
    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"创建了示例数据集: {output_dir}")
    print("注意: 这是示例数据，需要您添加真实的商品图像")
    return True

def main():
    """主函数"""
    print("替代数据集下载方案")
    print("=" * 40)
    
    print("\n由于大型数据集下载可能遇到问题，请选择:")
    print("1. 下载较小的Fashion-MNIST数据集（~30MB）")
    print("2. 创建示例数据集结构（用于测试）")
    print("3. 手动下载建议")
    
    choice = input("\n请输入选项 (1-3): ")
    
    if choice == '1':
        success = download_small_fashion_dataset()
        if success:
            print("\n下一步: 运行 python scripts/data_collection/dataset_builder.py")
    elif choice == '2':
        create_sample_dataset()
        print("\n下一步: 添加真实图像后运行 dataset_builder.py")
    elif choice == '3':
        print("\n手动下载建议:")
        print("1. 访问: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
        print("2. 手动下载ZIP文件")
        print("3. 解压到 data/raw/kaggle/fashion/ 目录")
        print("4. 运行: python scripts/data_collection/dataset_builder.py")
    else:
        print("无效选项")

if __name__ == "__main__":
    main()