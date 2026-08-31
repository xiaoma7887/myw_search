#!/usr/bin/env python3
"""
CLIP模型续训脚本
在电商图文对上续训Chinese CLIP模型，对齐电商场景
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class CLIPFineTuner:
    """CLIP模型续训器，实现部分冻结策略和分组学习率"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        
        # 加载预训练模型
        print("加载预训练模型...")
        self.model = ChineseCLIPModel.from_pretrained(model_config['name'])
        self.processor = ChineseCLIPProcessor.from_pretrained(model_config['name'])
        
        self.device = torch.device(model_config['device'])
        self.model.to(self.device)
        
        # 应用部分冻结策略
        self._freeze_parameters()
        
        # 训练参数（根据需求）
        self.batch_size = 96
        self.num_epochs = 30
        self.train_val_split = 0.9  # 9:1划分
        self.grad_clip_norm = 1.0
        self.early_stop_patience = 5
        
        # 分组学习率（根据需求）
        self.learning_rates = {
            'transformer': 1.5e-5,
            'normalization': 1.5e-5,
            'projection': 3e-4,
            'temperature': 3e-6
        }
        
        # 分组权重衰减（根据需求）
        self.weight_decays = {
            'transformer': 0.02,
            'normalization': 0.0,
            'projection': 0.02,
            'temperature': 0.0
        }
        
        # 创建优化器（分组参数）
        self.optimizer = self._create_grouped_optimizer()
    
    def _freeze_parameters(self):
        """
        应用部分冻结策略：
        - 冻结嵌入层
        - 文本模型：冻结前9个Transformer Block，解冻后3个
        - 图像模型：冻结前10个Transformer Block，解冻后2个
        - 解冻归一化权重、投影层、温度缩放权重
        """
        print("应用部分冻结策略...")
        
        # 冻结所有嵌入层
        if hasattr(self.model, 'text_model') and hasattr(self.model.text_model, 'embeddings'):
            for param in self.model.text_model.embeddings.parameters():
                param.requires_grad = False
        
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'embeddings'):
            for param in self.model.vision_model.embeddings.parameters():
                param.requires_grad = False
        
        # 文本模型：冻结前9个Transformer Block，解冻后3个
        if hasattr(self.model, 'text_model') and hasattr(self.model.text_model, 'encoder'):
            encoder = self.model.text_model.encoder
            if hasattr(encoder, 'layer'):
                layers = encoder.layer
                total_layers = len(layers)
                freeze_layers = total_layers - 3  # 保留最后3层
                
                for i, layer in enumerate(layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
        
        # 图像模型：冻结前10个Transformer Block，解冻后2个
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
            encoder = self.model.vision_model.encoder
            if hasattr(encoder, 'layer'):
                layers = encoder.layer
                total_layers = len(layers)
                freeze_layers = total_layers - 2  # 保留最后2层
                
                for i, layer in enumerate(layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
        
        # 解冻归一化层
        for module in self.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                for param in module.parameters():
                    param.requires_grad = True
        
        # 解冻投影层
        if hasattr(self.model, 'visual_projection'):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = True
        if hasattr(self.model, 'text_projection'):
            for param in self.model.text_projection.parameters():
                param.requires_grad = True
        
        # 解冻温度缩放参数（如果存在）
        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def _create_grouped_optimizer(self):
        """创建分组学习率和权重衰减的优化器"""
        # 分组参数
        param_groups = {
            'transformer': [],
            'normalization': [],
            'projection': [],
            'temperature': []
        }
        
        # 分类参数
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'encoder.layer' in name:
                param_groups['transformer'].append(param)
            elif 'norm' in name.lower() or 'ln' in name.lower():
                param_groups['normalization'].append(param)
            elif 'projection' in name.lower():
                param_groups['projection'].append(param)
            elif 'logit_scale' in name or 'temperature' in name.lower():
                param_groups['temperature'].append(param)
            else:
                # 默认归为transformer
                param_groups['transformer'].append(param)
        
        # 构建优化器参数组
        optimizer_groups = []
        for group_name, params in param_groups.items():
            if params:
                optimizer_groups.append({
                    'params': params,
                    'lr': self.learning_rates[group_name],
                    'weight_decay': self.weight_decays[group_name]
                })
                print(f"{group_name}: {len(params)} 个参数组, lr={self.learning_rates[group_name]}, wd={self.weight_decays[group_name]}")
        
        return torch.optim.AdamW(optimizer_groups)
    
    def prepare_dataset(self, data_file: str):
        """准备训练数据集"""
        print(f"加载数据文件: {data_file}")
        
        # 支持JSON格式的数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取文本和图像路径
        texts = []
        image_paths = []
        
        for item in data:
            if isinstance(item, dict):
                texts.append(item.get('text', ''))
                image_paths.append(item.get('image_path', ''))
            else:
                # 如果是其他格式，需要适配
                texts.append(str(item))
                image_paths.append(str(item))
        
        # 9:1划分训练集和验证集
        train_texts, val_texts, train_images, val_images = train_test_split(
            texts, image_paths, test_size=1-self.train_val_split, random_state=42
        )
        
        return (train_texts, train_images), (val_texts, val_images)
    
    def collate_fn(self, batch):
        """数据整理函数"""
        texts, image_paths = zip(*batch)
        
        # 处理文本
        text_inputs = self.processor(
            text=list(texts), 
            return_tensors="pt", 
            padding=True,
            max_length=self.config['processing']['text']['max_length'],
            truncation=True
        )
        
        # 处理图像
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"警告: 无法加载图像 {path}: {e}")
                # 使用占位符
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        
        return {
            'text_inputs': text_inputs,
            'image_inputs': image_inputs
        }
    
    def train_epoch(self, dataloader, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="训练中"):
            # 将数据移动到设备
            text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
            image_inputs = {k: v.to(self.device) for k, v in batch['image_inputs'].items()}
            
            # 前向传播
            outputs = self.model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                pixel_values=image_inputs['pixel_values']
            )
            
            # 计算对比损失
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # 创建标签（对角线匹配）
            batch_size = logits_per_image.shape[0]
            labels = torch.arange(batch_size, device=self.device)
            
            # 对称损失
            loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证中"):
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                image_inputs = {k: v.to(self.device) for k, v in batch['image_inputs'].items()}
                
                outputs = self.model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    pixel_values=image_inputs['pixel_values']
                )
                
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                batch_size = logits_per_image.shape[0]
                labels = torch.arange(batch_size, device=self.device)
                
                loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_dataset, val_dataset):
        """完整的训练流程"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4
        )
        
        # 计算总步数和预热步数（前10%步数预热）
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * 0.1)
        
        print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
        
        # 创建学习率调度器（余弦衰减，最小值为最大值的10%）
        # 注意：由于使用分组学习率，需要为每个参数组单独设置调度器
        # 这里使用自定义的调度逻辑
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 线性预热：从0到1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 余弦衰减：从1到0.1（最大值的10%）
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.1 + 0.9 * (1.0 + np.cos(np.pi * progress)) / 2.0
        
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"开始训练，共 {self.num_epochs} 个epoch...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader, scheduler)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"✓ 验证损失改善，保存最佳模型")
            else:
                patience_counter += 1
                print(f"验证损失未改善 ({patience_counter}/{self.early_stop_patience})")
                
                if patience_counter >= self.early_stop_patience:
                    print(f"早停触发！最佳验证损失: {best_val_loss:.4f}")
                    if best_model_state:
                        self.model.load_state_dict(best_model_state)
                    break
        
        return best_val_loss
    
    def save_model(self, output_dir: str):
        """保存微调后的模型"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"模型已保存到: {output_dir}")


class ImageTextDataset(Dataset):
    """图像-文本对数据集"""
    
    def __init__(self, texts, image_paths):
        self.texts = texts
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.image_paths[idx]


def main():
    parser = argparse.ArgumentParser(description='续训CLIP模型')
    parser.add_argument('--data_file', type=str, required=True,
                       help='训练数据文件路径（JSON格式）')
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_model',
                       help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数（默认30）')
    parser.add_argument('--batch_size', type=int, default=96,
                       help='批次大小（默认96）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在: {args.data_file}")
        return
    
    # 初始化训练器
    trainer = CLIPFineTuner()
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    
    # 准备数据
    print("准备数据集...")
    (train_texts, train_images), (val_texts, val_images) = trainer.prepare_dataset(args.data_file)
    
    print(f"训练集: {len(train_texts)} 对, 验证集: {len(val_texts)} 对")
    
    train_dataset = ImageTextDataset(train_texts, train_images)
    val_dataset = ImageTextDataset(val_texts, val_images)
    
    # 开始训练
    best_val_loss = trainer.train(train_dataset, val_dataset)
    
    # 保存模型
    trainer.save_model(args.output_dir)
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
