import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Union
import yaml


class ImageProcessor:
    """图像处理器，用于图像预处理和增强"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化图像处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        processing_config = self.config['processing']['image']
        
        self.target_size = tuple(processing_config['target_size'])
        self.mean = np.array(processing_config['mean']).reshape(1, 1, 3)
        self.std = np.array(processing_config['std']).reshape(1, 1, 3)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            RGB格式的图像数组
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 使用OpenCV加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        预处理单张图像
        
        Args:
            image: 图像路径或图像数组
            
        Returns:
            预处理后的图像数组
        """
        if isinstance(image, str):
            image = self.load_image(image)
        
        # 调整尺寸
        image_resized = cv2.resize(image, self.target_size)
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_normalized - self.mean) / self.std
        
        # 调整通道顺序为 (C, H, W)
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        return image_transposed
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        批量预处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            预处理后的图像批次，形状为 (batch_size, C, H, W)
        """
        processed_images = []
        
        for image_path in image_paths:
            try:
                processed_image = self.preprocess_image(image_path)
                processed_images.append(processed_image)
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
        
        return np.array(processed_images)
    
    def validate_image(self, image_path: str) -> bool:
        """
        验证图像文件是否有效
        
        Args:
            image_path: 图像路径
            
        Returns:
            是否有效
        """
        try:
            image = self.load_image(image_path)
            return image is not None and image.size > 0
        except:
            return False
    
    def get_image_info(self, image_path: str) -> dict:
        """
        获取图像信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像信息字典
        """
        try:
            image = self.load_image(image_path)
            return {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2],
                'file_size': os.path.getsize(image_path)
            }
        except Exception as e:
            return {'error': str(e)}