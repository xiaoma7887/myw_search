import torch
import yaml
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from typing import Union, List
import numpy as np
from PIL import Image


class CLIPModelWrapper:
    """CLIP模型封装类，用于处理图像和文本的特征提取"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化CLIP模型
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        
        # 加载模型和处理器
        self.model_name = model_config['name']
        self.device = torch.device(model_config['device'])
        
        self.model = ChineseCLIPModel.from_pretrained(self.model_name)
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 模型维度配置
        self.vision_output_dim = model_config['vision_model_output_dim']
        self.text_output_dim = model_config['text_model_output_dim']
        self.projection_dim = model_config['projection_dim']
        
    def get_image_embedding(self, images: Union[str, List[str]]) -> np.ndarray:
        """
        获取图像的特征向量
        
        Args:
            images: 图像路径或图像路径列表
            
        Returns:
            numpy数组，形状为 (n_images, projection_dim)
        """
        if isinstance(images, str):
            images = [images]
        
        # 加载图像为PIL Image对象
        pil_images = []
        for img_path in images:
            try:
                pil_img = Image.open(img_path).convert('RGB')
                pil_images.append(pil_img)
            except Exception as e:
                print(f"警告: 无法加载图像 {img_path}: {e}")
                continue
        
        if not pil_images:
            raise ValueError("没有有效的图像可以处理")
            
        # 预处理图像
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            # 使用 pooler_output（如果存在且非None），否则使用CLS token
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                image_embeds = vision_outputs.pooler_output
            else:
                # 使用CLS token（第一个token）
                image_embeds = vision_outputs.last_hidden_state[:, 0, :]
            image_embeds = self.model.visual_projection(image_embeds)
            
            # L2归一化（与需求一致：归一化后写入向量数据库）
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
        return image_embeds.cpu().numpy()
    
    def get_text_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        获取文本的特征向量
        
        Args:
            texts: 文本字符串或文本列表
            
        Returns:
            numpy数组，形状为 (n_texts, projection_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 预处理文本
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, 
                              max_length=self.config['processing']['text']['max_length'],
                              truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            # 使用 pooler_output（如果存在且非None），否则使用CLS token
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_embeds = text_outputs.pooler_output
            else:
                # 使用CLS token（第一个token）
                text_embeds = text_outputs.last_hidden_state[:, 0, :]
            text_embeds = self.model.text_projection(text_embeds)
            
            # L2归一化（与需求一致：归一化后用于检索）
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
        return text_embeds.cpu().numpy()
    
    def get_embeddings(self, images: Union[str, List[str]] = None, 
                      texts: Union[str, List[str]] = None) -> dict:
        """
        同时获取图像和文本的特征向量
        
        Args:
            images: 图像路径或图像路径列表
            texts: 文本字符串或文本列表
            
        Returns:
            包含图像和文本向量的字典
        """
        result = {}
        
        if images is not None:
            result['image_embeddings'] = self.get_image_embedding(images)
            
        if texts is not None:
            result['text_embeddings'] = self.get_text_embedding(texts)
            
        return result