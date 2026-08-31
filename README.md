# 语义商品搜索系统

基于CLIP和Chroma的语义商品搜索系统，能够根据用户输入的文本查询，在电商场景下搜索相似的商品图像。

## 项目概述

本项目使用预训练的Chinese CLIP模型，在电商图文对上进行了续训，能够将图像和查询文本映射到同一向量空间，通过向量相似度检索相似商品。

## 核心功能

- 图像特征提取：使用CLIP vision_model处理图像生成向量表征
- 文本特征提取：使用CLIP text_model处理查询文本生成向量表征  
- 向量存储：使用Chroma向量数据库存储和管理图像向量
- 语义搜索：根据文本查询检索最相似的图像

## 技术栈

- **模型框架**: Chinese CLIP (OFA-Sys/chinese-clip-vit-base-patch16)
- **向量数据库**: Chroma
- **编程语言**: Python
- **深度学习框架**: PyTorch / Transformers

## 项目结构

```
semantic-product-search/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_model.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── chroma_db.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── image_processor.py
│   │   └── text_processor.py
│   └── search/
│       ├── __init__.py
│       └── semantic_search.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── setup_database.py
│   └── train_model.py
├── tests/
├── requirements.txt
└── main.py
```

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 配置模型和数据库参数：修改 `config/config.yaml`
3. 初始化向量数据库：运行 `python scripts/setup_database.py`
4. 启动搜索服务：运行 `python main.py`

## 使用说明

系统支持两种主要使用方式：
1. **批量处理**: 将商品图像批量导入向量数据库
2. **实时搜索**: 接收用户文本查询，返回相似商品图像