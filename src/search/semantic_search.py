import yaml
import numpy as np
from typing import List, Dict, Any
from ..models.clip_model import CLIPModelWrapper
from ..database.chroma_db import ChromaDBManager
from ..processing.text_processor import TextProcessor


class SemanticSearchEngine:
    """语义搜索引擎，整合CLIP模型和向量数据库"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化语义搜索引擎
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化各组件
        self.clip_model = CLIPModelWrapper(config_path)
        self.db_manager = ChromaDBManager(config_path)
        self.text_processor = TextProcessor(config_path)
        
        self.top_k = self.config['search']['top_k']
        self.similarity_threshold = self.config['search']['similarity_threshold']
    
    def add_images_to_index(self, image_paths: List[str], 
                          metadatas: List[Dict] = None) -> Dict[str, Any]:
        """
        将图像添加到搜索索引
        
        Args:
            image_paths: 图像路径列表
            metadatas: 元数据列表
            
        Returns:
            添加结果统计
        """
        # 验证图像
        valid_paths = []
        valid_metadatas = []
        
        for i, path in enumerate(image_paths):
            try:
                # 这里可以添加图像验证逻辑
                valid_paths.append(path)
                if metadatas and i < len(metadatas):
                    valid_metadatas.append(metadatas[i])
                else:
                    valid_metadatas.append({"image_path": path})
            except Exception as e:
                print(f"跳过无效图像 {path}: {e}")
                continue
        
        if not valid_paths:
            return {"success": False, "message": "没有有效的图像可添加"}
        
        # 提取图像特征
        print(f"正在提取 {len(valid_paths)} 张图像的特征...")
        embeddings = self.clip_model.get_image_embedding(valid_paths)
        
        # 添加到数据库
        print("正在将特征向量添加到数据库...")
        self.db_manager.add_images(valid_paths, embeddings, valid_metadatas)
        
        return {
            "success": True,
            "added_count": len(valid_paths),
            "total_count": self.db_manager.get_collection_info()['count']
        }
    
    def search_by_text(self, query_text: str, top_k: int = None) -> Dict[str, Any]:
        """
        根据文本查询搜索相似图像
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        if top_k is None:
            top_k = self.top_k
        
        # 预处理查询文本
        processed_text = self.text_processor.preprocess_text(query_text)
        
        # 提取文本特征
        text_embedding = self.clip_model.get_text_embedding(processed_text)
        
        # 在数据库中搜索
        search_results = self.db_manager.search_by_text(text_embedding, top_k)
        
        # 过滤相似度阈值
        filtered_results = self._filter_by_similarity(search_results)
        
        # 计算相似度用于显示
        similarities = []
        if search_results.get('distances'):
            similarities = [1 - d for d in search_results['distances']]
        
        return {
            "query": query_text,
            "processed_query": processed_text,
            "total_results": len(search_results['image_paths']),
            "filtered_results": len(filtered_results['image_paths']),
            "raw_similarities": similarities[:5] if similarities else [],  # 显示原始相似度
            "results": filtered_results
        }
    
    def search_by_image(self, image_path: str, top_k: int = None) -> Dict[str, Any]:
        """
        根据图像搜索相似图像
        
        Args:
            image_path: 查询图像路径
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        if top_k is None:
            top_k = self.top_k
        
        # 提取图像特征
        image_embedding = self.clip_model.get_image_embedding(image_path)
        
        # 在数据库中搜索
        search_results = self.db_manager.search_by_image(image_embedding, top_k)
        
        # 过滤相似度阈值
        filtered_results = self._filter_by_similarity(search_results)
        
        return {
            "query_image": image_path,
            "total_results": len(search_results['image_paths']),
            "filtered_results": len(filtered_results['image_paths']),
            "results": filtered_results
        }
    
    def _filter_by_similarity(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据相似度阈值过滤结果
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            过滤后的结果
        """
        if not search_results['distances']:
            return search_results
        
        # 余弦距离转换为相似度 (1 - distance)
        similarities = [1 - distance for distance in search_results['distances']]
        
        # 过滤
        filtered_indices = [
            i for i, similarity in enumerate(similarities) 
            if similarity >= self.similarity_threshold
        ]
        
        return {
            'distances': [search_results['distances'][i] for i in filtered_indices],
            'similarities': [similarities[i] for i in filtered_indices],
            'image_paths': [search_results['image_paths'][i] for i in filtered_indices],
            'ids': [search_results['ids'][i] for i in filtered_indices]
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        db_info = self.db_manager.get_collection_info()
        
        return {
            "database": {
                "collection_name": db_info['name'],
                "total_images": db_info['count'],
                "embedding_dim": self.config['model']['projection_dim']
            },
            "model": {
                "name": self.config['model']['name'],
                "device": self.config['model']['device']
            },
            "search": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            }
        }
    
    def reset_index(self) -> Dict[str, Any]:
        """
        重置搜索索引
        
        Returns:
            重置结果
        """
        self.db_manager.reset_collection()
        return {"success": True, "message": "索引已重置"}