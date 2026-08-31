#!/usr/bin/env python3
"""调试模型输出结构"""
from src.models.clip_model import CLIPModelWrapper
from PIL import Image
import os

# 初始化模型
model = CLIPModelWrapper()

# 测试图像
test_image = "data/raw/kaggle/fashion/fashion-dataset/images/10000.jpg"
if os.path.exists(test_image):
    print("测试图像特征提取...")
    img = Image.open(test_image).convert('RGB')
    
    # 测试 vision_model 输出
    from transformers import ChineseCLIPProcessor
    processor = ChineseCLIPProcessor.from_pretrained(model.model_name)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    import torch
    with torch.no_grad():
        vision_outputs = model.model.vision_model(**inputs)
        print(f"vision_outputs 类型: {type(vision_outputs)}")
        if isinstance(vision_outputs, tuple):
            print(f"vision_outputs 长度: {len(vision_outputs)}")
            for i, v in enumerate(vision_outputs):
                print(f"  [{i}] 类型: {type(v)}, 形状: {v.shape if hasattr(v, 'shape') else 'N/A'}")
        else:
            print(f"vision_outputs 属性: {dir(vision_outputs)}")
            if hasattr(vision_outputs, 'last_hidden_state'):
                print(f"last_hidden_state 形状: {vision_outputs.last_hidden_state.shape}")
            if hasattr(vision_outputs, 'pooler_output'):
                print(f"pooler_output: {vision_outputs.pooler_output}")
    
    # 测试文本模型
    text_inputs = processor(text="测试", return_tensors="pt")
    text_inputs = {k: v.to(model.device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_outputs = model.model.text_model(**text_inputs)
        print(f"\ntext_outputs 类型: {type(text_outputs)}")
        if isinstance(text_outputs, tuple):
            print(f"text_outputs 长度: {len(text_outputs)}")
            for i, v in enumerate(text_outputs):
                print(f"  [{i}] 类型: {type(v)}, 形状: {v.shape if hasattr(v, 'shape') else 'N/A'}")
        else:
            print(f"text_outputs 属性: {dir(text_outputs)}")
            if hasattr(text_outputs, 'last_hidden_state'):
                print(f"last_hidden_state 形状: {text_outputs.last_hidden_state.shape}")
            if hasattr(text_outputs, 'pooler_output'):
                print(f"pooler_output: {text_outputs.pooler_output}")
else:
    print(f"测试图像不存在: {test_image}")

