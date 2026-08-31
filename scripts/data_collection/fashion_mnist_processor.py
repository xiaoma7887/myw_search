#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion-MNIST数据集处理器
将Fashion-MNIST二进制数据转换为CLIP训练格式
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
import gzip
import numpy as np
from PIL import Image
import json
import random

class FashionMNISTProcessor:
    """Fashion-MNIST数据处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.labels_map = {
            0: "T恤/上衣",
            1: "裤子", 
            2: "套头衫",
            3: "连衣裙",
            4: "外套",
            5: "凉鞋",
            6: "衬衫",
            7: "运动鞋",
            8: "包",
            9: "短靴"
        }
    
    def load_mnist_images(self, filename: str) -> np.ndarray:
        """
        加载MNIST图像数据
        
        Args:
            filename: 图像文件路径
            
        Returns:
            图像数组
        """
        with gzip.open(filename, 'rb') as f:
            # 读取魔数
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2051:
                raise ValueError(f"无效的MNIST图像文件: {filename}")
            
            # 读取图像数量、行数、列数
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # 读取图像数据
            buffer = f.read(rows * cols * num_images)
            data = np.frombuffer(buffer, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            
            return data
    
    def load_mnist_labels(self, filename: str) -> np.ndarray:
        """
        加载MNIST标签数据
        
        Args:
            filename: 标签文件路径
            
        Returns:
            标签数组
        """
        with gzip.open(filename, 'rb') as f:
            # 读取魔数
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2049:
                raise ValueError(f"无效的MNIST标签文件: {filename}")
            
            # 读取标签数量
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # 读取标签数据
            buffer = f.read(num_labels)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            
            return labels
    
    def save_images_as_png(self, images: np.ndarray, labels: np.ndarray, 
                          output_dir: str, max_images: int = 1000) -> list:
        """
        将图像保存为PNG格式
        
        Args:
            images: 图像数组
            labels: 标签数组
            output_dir: 输出目录
            max_images: 最大保存图像数量
            
        Returns:
            图像信息列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_info = []
        
        # 限制处理数量
        num_images = min(len(images), max_images)
        
        for i in range(num_images):
            image = images[i]
            label = labels[i]
            
            # 转换为PIL图像
            img_pil = Image.fromarray(image, mode='L')
            
            # 转换为RGB（CLIP需要3通道）
            img_rgb = img_pil.convert('RGB')
            
            # 调整尺寸为224x224（CLIP输入尺寸）
            img_resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 保存图像
            filename = f"fashion_{i:05d}.png"
            filepath = os.path.join(output_dir, filename)
            img_resized.save(filepath, 'PNG')
            
            # 获取标签文本
            label_text = self.labels_map.get(label, "未知")
            
            # 构建描述文本
            descriptions = [
                f"{label_text} 时尚单品",
                f"{label_text} 商品图像", 
                f"时尚 {label_text} 图片",
                f"{label_text} 服装款式"
            ]
            
            # 随机选择一个描述
            description = random.choice(descriptions)
            
            image_info.append({
                'image_path': filepath,
                'text': description,
                'label': label,
                'label_text': label_text,
                'source': 'fashion_mnist'
            })
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{num_images} 张图像")
        
        return image_info
    
    def create_training_dataset(self, raw_data_dir: str, 
                              output_dir: str = "data/processed/clip_training") -> bool:
        """
        创建训练数据集
        
        Args:
            raw_data_dir: 原始数据目录
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            print("处理Fashion-MNIST数据集...")
            
            # 加载训练数据
            images_path = os.path.join(raw_data_dir, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(raw_data_dir, "train-labels-idx1-ubyte.gz")
            
            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                print("错误: 找不到Fashion-MNIST数据文件")
                return False
            
            print("加载图像数据...")
            images = self.load_mnist_images(images_path)
            
            print("加载标签数据...")
            labels = self.load_mnist_labels(labels_path)
            
            print(f"加载了 {len(images)} 张图像和 {len(labels)} 个标签")
            
            # 保存为PNG格式
            images_output_dir = os.path.join(raw_data_dir, "images")
            print("转换图像格式...")
            
            image_info = self.save_images_as_png(
                images, labels, images_output_dir, max_images=1000
            )
            
            print(f"成功转换 {len(image_info)} 张图像")
            
            # 分割数据集
            print("分割数据集...")
            random.shuffle(image_info)
            
            total_count = len(image_info)
            train_count = int(total_count * 0.8)
            val_count = int(total_count * 0.1)
            
            dataset = {
                'train': image_info[:train_count],
                'val': image_info[train_count:train_count + val_count],
                'test': image_info[train_count + val_count:]
            }
            
            # 保存数据集
            print("保存数据集...")
            os.makedirs(output_dir, exist_ok=True)
            
            for split_name, split_data in dataset.items():
                # 保存JSON文件
                json_path = os.path.join(output_dir, f"{split_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, ensure_ascii=False, indent=2)
                
                # 保存文本文件
                txt_path = os.path.join(output_dir, f"{split_name}.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(f"{item['image_path']}\t{item['text']}\n")
                
                print(f"{split_name}集: {len(split_data)} 个样本")
            
            # 保存统计信息
            stats = {
                'total_samples': total_count,
                'train_samples': len(dataset['train']),
                'val_samples': len(dataset['val']),
                'test_samples': len(dataset['test']),
                'categories': len(set(item['label_text'] for split in dataset.values() for item in split))
            }
            
            stats_path = os.path.join(output_dir, "dataset_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"\n数据集构建完成!")
            print(f"总样本数: {stats['total_samples']}")
            print(f"类别数: {stats['categories']}")
            print(f"输出目录: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"处理失败: {e}")
            return False

def main():
    """主函数"""
    processor = FashionMNISTProcessor()
    
    success = processor.create_training_dataset(
        raw_data_dir="data/raw/fashion_mnist",
        output_dir="data/processed/clip_training"
    )
    
    if success:
        print("\n🎉 Fashion-MNIST数据集处理完成!")
        print("下一步: 可以使用这个数据集来训练CLIP模型了")
    else:
        print("\n❌ 数据集处理失败")

if __name__ == "__main__":
    main()