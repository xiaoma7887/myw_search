import re
import jieba
from typing import List, Union
import yaml


class TextProcessor:
    """文本处理器，用于文本预处理和清洗"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化文本处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.max_length = self.config['processing']['text']['max_length']
        
        # 初始化分词器
        jieba.initialize()
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 移除特殊字符和多余空格
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        分词处理
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的列表
        """
        return list(jieba.cut(text))
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理单条文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 长度限制
        if len(cleaned_text) > self.max_length:
            cleaned_text = cleaned_text[:self.max_length]
        
        return cleaned_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            预处理后的文本列表
        """
        return [self.preprocess_text(text) for text in texts]
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表
        """
        tokens = self.tokenize_text(text)
        
        # 简单的词频统计（实际应用中可以使用TF-IDF等更复杂的方法）
        word_freq = {}
        for token in tokens:
            if len(token) > 1:  # 过滤单字
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:top_k]]
    
    def validate_text(self, text: str) -> bool:
        """
        验证文本是否有效
        
        Args:
            text: 输入文本
            
        Returns:
            是否有效
        """
        if not text or not text.strip():
            return False
        
        cleaned_text = self.clean_text(text)
        return len(cleaned_text) > 0
    
    def get_text_info(self, text: str) -> dict:
        """
        获取文本信息
        
        Args:
            text: 输入文本
            
        Returns:
            文本信息字典
        """
        cleaned_text = self.preprocess_text(text)
        tokens = self.tokenize_text(cleaned_text)
        keywords = self.extract_keywords(cleaned_text)
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned_text),
            'token_count': len(tokens),
            'keywords': keywords
        }