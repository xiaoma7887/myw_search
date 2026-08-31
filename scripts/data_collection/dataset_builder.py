#!/usr/bin/env python3
"""
数据集构建工具
用于处理采集的淘宝数据，构建CLIP训练数据集
"""

import os
import json
import yaml
import random
from typing import List, Dict, Any
from PIL import Image
import numpy as np


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, config_path: str = "config/taobao_config.yaml"):
        """
        初始化数据集构建器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
    
    def load_raw_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        加载原始数据
        
        Args:
            data_dir: 数据目录
            
        Returns:
            原始数据列表
        """
        data_file = os.path.join(data_dir, "taobao_dataset.json")
        
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            return []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        text_config = self.dataset_config['preprocessing']['text']
        
        # 移除特殊字符
        if text_config['remove_special_chars']:
            import re
            text = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        # 长度限制
        if len(text) < text_config['min_length']:
            return ""
        
        if len(text) > text_config['max_length']:
            text = text[:text_config['max_length']]
        
        return text
    
    def validate_image(self, image_path: str) -> bool:
        """
        验证图像文件
        
        Args:
            image_path: 图像路径
            
        Returns:
            是否有效
        """
        image_config = self.dataset_config['preprocessing']['image']
        
        try:
            # 检查文件存在
            if not os.path.exists(image_path):
                return False
            
            # 检查文件大小
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > image_config['max_file_size_mb']:
                return False
            
            # 检查图像格式
            file_ext = os.path.splitext(image_path)[1].lower().lstrip('.')
            if file_ext not in image_config['supported_formats']:
                return False
            
            # 检查图像尺寸
            with Image.open(image_path) as img:
                width, height = img.size
                if width < image_config['min_width'] or height < image_config['min_height']:
                    return False
                
                # 检查图像完整性
                img.verify()
            
            return True
            
        except Exception:
            return False
    
    def build_training_pairs(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建训练数据对
        
        Args:
            raw_data: 原始数据
            
        Returns:
            训练数据对列表
        """
        training_pairs = []
        
        for product in raw_data:
            # 验证图像
            if not self.validate_image(product['image_path']):
                continue
            
            # 预处理文本
            processed_title = self.preprocess_text(product['title'])
            processed_desc = self.preprocess_text(product.get('description', ''))
            
            # 使用标题作为主要文本
            if processed_title:
                training_pairs.append({
                    'image_path': product['image_path'],
                    'text': processed_title,
                    'product_id': product['product_id'],
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'source': 'taobao'
                })
            
            # 如果描述可用且与标题不同，也作为训练对
            if processed_desc and processed_desc != processed_title:
                training_pairs.append({
                    'image_path': product['image_path'],
                    'text': processed_desc,
                    'product_id': product['product_id'],
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'source': 'taobao'
                })
        
        # 去重
        if self.dataset_config['preprocessing']['text']['remove_duplicates']:
            training_pairs = self._remove_duplicates(training_pairs)
        
        return training_pairs
    
    def _remove_duplicates(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去除重复数据
        """
        seen = set()
        unique_pairs = []
        
        for pair in pairs:
            # 基于图像路径和文本内容的组合来判断重复
            key = (pair['image_path'], pair['text'])
            
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)
        
        return unique_pairs
    
    def split_dataset(self, pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        分割数据集
        
        Args:
            pairs: 数据对列表
            
        Returns:
            分割后的数据集
        """
        split_ratio = self.dataset_config['split_ratio']
        
        # 随机打乱
        random.shuffle(pairs)
        
        total_count = len(pairs)
        train_count = int(total_count * split_ratio['train'])
        val_count = int(total_count * split_ratio['val'])
        
        return {
            'train': pairs[:train_count],
            'val': pairs[train_count:train_count + val_count],
            'test': pairs[train_count + val_count:]
        }
    
    def save_dataset(self, dataset: Dict[str, List[Dict[str, Any]]], output_dir: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            # 保存JSON文件
            json_path = os.path.join(output_dir, f"{split_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # 保存文本文件（每行一个图像路径和文本）
            txt_path = os.path.join(output_dir, f"{split_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(f"{item['image_path']}\t{item['text']}\n")
            
            print(f"{split_name}集: {len(split_data)} 个样本")
        
        # 保存数据集统计信息
        stats = {
            'total_samples': sum(len(split) for split in dataset.values()),
            'train_samples': len(dataset['train']),
            'val_samples': len(dataset['val']),
            'test_samples': len(dataset['test']),
            'categories': len(set(item['category'] for split in dataset.values() for item in split if item['category']))
        }
        
        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到: {output_dir}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别数: {stats['categories']}")
    
    def build_dataset(self, raw_data_dir: str, output_dir: str = "data/processed/clip_training"):
        """
        构建完整数据集
        
        Args:
            raw_data_dir: 原始数据目录
            output_dir: 输出目录
        """
        print("加载原始数据...")
        raw_data = self.load_raw_data(raw_data_dir)
        
        if not raw_data:
            print("没有找到原始数据")
            return
        
        print(f"加载了 {len(raw_data)} 个原始商品")
        
        print("构建训练数据对...")
        training_pairs = self.build_training_pairs(raw_data)
        
        print(f"生成了 {len(training_pairs)} 个训练数据对")
        
        print("分割数据集...")
        dataset = self.split_dataset(training_pairs)
        
        print("保存数据集...")
        self.save_dataset(dataset, output_dir)


def main():
    """主函数"""
    builder = DatasetBuilder()
    
    # 构建数据集
    builder.build_dataset(
        raw_data_dir="data/raw/taobao",
        output_dir="data/processed/clip_training"
    )


if __name__ == "__main__":
    main()