# ⚠️ 重要提示：需要重建数据库

## 为什么需要重建？

由于我们修改了两个关键配置：
1. **距离度量**：从 `cosine` 改为 `ip`（点积）
2. **向量归一化**：添加了L2归一化

这两个改变会影响向量相似度的计算方式，因此需要重建数据库。

## 重建步骤

### 方法1：删除旧数据库（推荐）

```bash
# 1. 停止服务（如果正在运行）
# 在运行服务的终端按 Ctrl+C

# 2. 删除旧数据库
# Windows PowerShell:
Remove-Item -Recurse -Force data\chroma_db

# 或 Linux/Mac:
# rm -rf data/chroma_db

# 3. 重新添加图像（使用新的归一化向量和ip距离）
python quick_setup.py 500

# 4. 重启服务
python main.py
```

### 方法2：使用脚本重置

```bash
# 使用API重置（如果服务正在运行）
python -c "import requests; r = requests.delete('http://localhost:9000/index/reset'); print(r.json())"

# 然后重新添加图像
python quick_setup.py 500
```

## 验证修复

重建后，验证以下内容：

1. **检查距离度量**：
```python
from src.database.chroma_db import ChromaDBManager
db = ChromaDBManager()
print(db.collection.metadata)  # 应该显示 "hnsw:space": "ip"
```

2. **检查向量归一化**：
```python
from src.models.clip_model import CLIPModelWrapper
model = CLIPModelWrapper()
emb = model.get_image_embedding("data/raw/kaggle/fashion/fashion-dataset/images/10000.jpg")
print(f"向量范数: {np.linalg.norm(emb)}")  # 应该接近 1.0
```

3. **测试搜索**：
```bash
python test_real_search.py
```

## 注意事项

- 重建数据库会丢失所有已添加的图像
- 确保有足够的图像数据可以重新添加
- 重建后需要重新运行 `quick_setup.py` 或 `scripts/setup_database.py`

