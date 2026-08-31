import chromadb
import yaml
import numpy as np
from typing import List, Dict, Any, Optional
import os
import hashlib


class ChromaDBManager:
    """Chroma向量数据库管理类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化Chroma数据库
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config['database']['chroma']
        
        # 创建持久化目录
        os.makedirs(db_config['persist_directory'], exist_ok=True)
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=db_config['persist_directory'])
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=db_config['collection_name'],
            metadata={"hnsw:space": db_config['distance_metric']}
        )
        
        self.embedding_dim = self.config['model']['projection_dim']
    
    def add_images(self, image_paths: List[str], embeddings: np.ndarray, 
                   metadatas: Optional[List[Dict]] = None) -> None:
        """
        添加图像向量到数据库
        
        Args:
            image_paths: 图像路径列表
            embeddings: 图像特征向量，形状为 (n_images, embedding_dim)
            metadatas: 元数据列表，每个元素为字典
        """
        if len(image_paths) != len(embeddings):
            raise ValueError("图像路径数量与向量数量不匹配")
        
        # 生成唯一ID（基于图像路径的哈希值）
        ids = [hashlib.md5(path.encode('utf-8')).hexdigest() for path in image_paths]
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{"image_path": path} for path in image_paths]
        else:
            for i, metadata in enumerate(metadatas):
                metadata["image_path"] = image_paths[i]
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
    
    def search_by_text(self, text_embedding: np.ndarray, top_k: int = None) -> Dict[str, Any]:
        """
        根据文本向量搜索相似图像
        
        Args:
            text_embedding: 文本特征向量，形状为 (1, embedding_dim)
            top_k: 返回最相似的前k个结果
            
        Returns:
            搜索结果字典
        """
        if top_k is None:
            top_k = self.config['search']['top_k']
        
        results = self.collection.query(
            query_embeddings=text_embedding.tolist(),
            n_results=top_k
        )
        
        return {
            'distances': results['distances'][0],
            'image_paths': [metadata['image_path'] for metadata in results['metadatas'][0]],
            'ids': results['ids'][0]
        }
    
    def search_by_image(self, image_embedding: np.ndarray, top_k: int = None) -> Dict[str, Any]:
        """
        根据图像向量搜索相似图像
        
        Args:
            image_embedding: 图像特征向量，形状为 (1, embedding_dim)
            top_k: 返回最相似的前k个结果
            
        Returns:
            搜索结果字典
        """
        return self.search_by_text(image_embedding, top_k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        return {
            'count': self.collection.count(),
            'name': self.collection.name,
            'metadata': self.collection.metadata
        }
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        根据ID删除向量
        
        Args:
            ids: 要删除的向量ID列表
        """
        self.collection.delete(ids=ids)
    
    def reset_collection(self) -> None:
        """重置集合（删除所有数据）"""
        db_config = self.config['database']['chroma']
        self.client.delete_collection(db_config['collection_name'])
        self.collection = self.client.get_or_create_collection(
            name=db_config['collection_name'],
            metadata={"hnsw:space": db_config['distance_metric']}
        )