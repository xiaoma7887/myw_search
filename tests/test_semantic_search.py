#!/usr/bin/env python3
"""
语义搜索系统测试
"""

import unittest
import os
import tempfile
import numpy as np
from PIL import Image

from src.search.semantic_search import SemanticSearchEngine
from src.models.clip_model import CLIPModelWrapper
from src.database.chroma_db import ChromaDBManager


class TestSemanticSearch(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试图像
        self.test_image_paths = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
            path = os.path.join(self.temp_dir, f'test_image_{i}.jpg')
            img.save(path)
            self.test_image_paths.append(path)
        
        # 创建测试配置文件
        self.test_config = {
            'model': {
                'name': 'OFA-Sys/chinese-clip-vit-base-patch16',
                'vision_model_output_dim': 768,
                'text_model_output_dim': 768,
                'projection_dim': 512,
                'image_size': 224,
                'device': 'cpu'
            },
            'database': {
                'chroma': {
                    'persist_directory': os.path.join(self.temp_dir, 'chroma_db'),
                    'collection_name': 'test_collection',
                    'distance_metric': 'cosine'
                }
            },
            'processing': {
                'image': {
                    'target_size': [224, 224],
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                },
                'text': {
                    'max_length': 64
                }
            },
            'search': {
                'top_k': 5,
                'similarity_threshold': 0.5
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False
            }
        }
        
        # 保存测试配置
        import yaml
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_clip_model_initialization(self):
        """测试CLIP模型初始化"""
        model = CLIPModelWrapper(self.config_path)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.processor)
    
    def test_text_embedding_generation(self):
        """测试文本向量生成"""
        model = CLIPModelWrapper(self.config_path)
        
        # 测试单条文本
        text = "测试文本"
        embedding = model.get_text_embedding(text)
        self.assertEqual(embedding.shape, (1, 512))
        
        # 测试多条文本
        texts = ["文本1", "文本2"]
        embeddings = model.get_text_embedding(texts)
        self.assertEqual(embeddings.shape, (2, 512))
    
    def test_image_embedding_generation(self):
        """测试图像向量生成"""
        model = CLIPModelWrapper(self.config_path)
        
        # 测试单张图像
        embedding = model.get_image_embedding(self.test_image_paths[0])
        self.assertEqual(embedding.shape, (1, 512))
        
        # 测试多张图像
        embeddings = model.get_image_embedding(self.test_image_paths)
        self.assertEqual(embeddings.shape, (3, 512))
    
    def test_chroma_db_initialization(self):
        """测试Chroma数据库初始化"""
        db_manager = ChromaDBManager(self.config_path)
        info = db_manager.get_collection_info()
        self.assertEqual(info['name'], 'test_collection')
    
    def test_semantic_search_engine(self):
        """测试语义搜索引擎"""
        search_engine = SemanticSearchEngine(self.config_path)
        
        # 测试添加图像
        result = search_engine.add_images_to_index(self.test_image_paths)
        self.assertTrue(result['success'])
        self.assertEqual(result['added_count'], 3)
        
        # 测试文本搜索
        search_result = search_engine.search_by_text("测试查询")
        self.assertIn('results', search_result)
        self.assertIn('image_paths', search_result['results'])
        
        # 测试系统信息
        system_info = search_engine.get_system_info()
        self.assertIn('database', system_info)
        self.assertIn('model', system_info)


if __name__ == '__main__':
    unittest.main()