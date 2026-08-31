# 项目需求对比分析报告

## 需求满足情况总结

### ✅ 已满足的需求

1. **模型选型**
   - ✅ 使用 `OFA-Sys/chinese-clip-vit-base-patch16`
   - ✅ vision_model和text_model输出768维
   - ✅ 投影到512维（`projection_dim: 512`）
   - ✅ 图像尺寸224×224

2. **向量数据库选型**
   - ✅ 使用Chroma作为向量数据库
   - ✅ 支持持久化存储
   - ✅ 使用HNSW索引（Chroma默认）

3. **基本检索流程**
   - ✅ 图像通过vision_model处理为512维向量
   - ✅ 文本通过text_model处理为512维向量
   - ✅ 从Chroma中检索相似图像

### ❌ 不满足的需求

#### 1. **续训功能不完整**

**需求**：
- 在25万+图文对上续训
- 部分冻结策略（冻结嵌入层、部分Transformer Block）
- 分组学习率（Transformer Block: 1.5e-5, 投影层: 3e-4, 温度缩放: 3e-6）
- 学习率策略（线性预热、余弦衰减）
- 梯度裁剪（范数1.0）
- 早停机制（容忍度5个epoch）

**当前状态**：
- ❌ `scripts/train_model.py` 只有基础训练框架
- ❌ 没有实现部分冻结策略
- ❌ 没有实现分组学习率
- ❌ 没有实现学习率预热和衰减
- ❌ 没有实现梯度裁剪
- ❌ 没有实现早停机制

#### 2. **向量归一化缺失**

**需求**：
- 图像表征和文本表征都需要**归一化**后写入/查询

**当前状态**：
- ❌ `src/models/clip_model.py` 中`get_image_embedding`和`get_text_embedding`没有进行L2归一化
- ❌ 直接返回投影后的向量，未归一化

#### 3. **距离度量不匹配**

**需求**：
- Chroma使用**ip（点积）**算法计算距离

**当前状态**：
- ❌ `config/config.yaml` 中配置为 `distance_metric: "cosine"`
- ❌ 应该改为 `"ip"` 或 `"dotproduct"`

#### 4. **训练脚本细节缺失**

**需求**：
- 批次大小96
- 30个epoch
- 训练/验证集9:1划分
- AdamW优化器，分组权重衰减（Transformer和投影层0.02，其余0.0）

**当前状态**：
- ❌ 批次大小32（应为96）
- ❌ 10个epoch（应为30）
- ❌ 没有实现训练/验证集划分逻辑
- ❌ 没有实现分组权重衰减

## 需要修改的文件

### 1. `src/models/clip_model.py` - 添加归一化

需要在返回向量前进行L2归一化：

```python
def get_image_embedding(self, images: Union[str, List[str]]) -> np.ndarray:
    # ... 现有代码 ...
    image_embeds = self.model.visual_projection(image_embeds)
    
    # 添加归一化
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    return image_embeds.cpu().numpy()
```

### 2. `config/config.yaml` - 修改距离度量

```yaml
database:
  chroma:
    distance_metric: "ip"  # 改为点积
```

### 3. `scripts/train_model.py` - 完善续训逻辑

需要添加：
- 部分冻结策略
- 分组学习率和权重衰减
- 学习率调度（预热+余弦衰减）
- 梯度裁剪
- 早停机制

## 优先级建议

### 高优先级（影响核心功能）
1. ✅ 添加向量归一化（必需，否则检索效果差）
2. ✅ 修改距离度量为ip（与需求一致）

### 中优先级（影响训练效果）
3. ✅ 完善训练脚本的续训功能
4. ✅ 实现部分冻结策略
5. ✅ 实现分组学习率

### 低优先级（优化）
6. ✅ 实现早停机制
7. ✅ 实现学习率调度策略

