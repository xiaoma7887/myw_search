# 数据库重建和测试指南

## 为什么需要重建？

由于修改了以下关键配置：
1. **距离度量**：从 `cosine` 改为 `ip`（点积）
2. **向量归一化**：添加了L2归一化

这两个改变会影响向量相似度的计算方式，因此需要重建数据库。

---

## 步骤1：停止服务（如果正在运行）

```bash
# 方法1：在运行服务的终端按 Ctrl+C

# 方法2：通过进程ID停止
# Windows PowerShell:
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Where-Object {(Get-NetTCPConnection -LocalPort 9000 -ErrorAction SilentlyContinue)} | Stop-Process -Force

# 验证端口是否释放
netstat -ano | findstr :9000
```

---

## 步骤2：删除旧数据库

```bash
# Windows PowerShell:
Remove-Item -Recurse -Force data\chroma_db

# 验证删除成功
Test-Path data\chroma_db  # 应该返回 False
```

---

## 步骤3：激活conda环境

```bash
conda activate text2img

# 验证环境
python --version  # 应该是 Python 3.10.19
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 步骤4：验证修复后的代码

### 4.1 验证向量归一化

```bash
python -c "
from src.models.clip_model import CLIPModelWrapper
import numpy as np

model = CLIPModelWrapper()
# 测试图像向量归一化
emb = model.get_image_embedding('data/raw/kaggle/fashion/fashion-dataset/images/10000.jpg')
norm = np.linalg.norm(emb)
print(f'图像向量范数: {norm:.6f} (应该接近1.0)')

# 测试文本向量归一化
emb_text = model.get_text_embedding('测试文本')
norm_text = np.linalg.norm(emb_text)
print(f'文本向量范数: {norm_text:.6f} (应该接近1.0)')
"
```

**预期输出**：
- 图像向量范数: 约 1.000000
- 文本向量范数: 约 1.000000

### 4.2 验证距离度量配置

```bash
python -c "
import yaml
config = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
print('距离度量:', config['database']['chroma']['distance_metric'])
# 应该输出: 距离度量: ip
"
```

---

## 步骤5：重建数据库

### 5.1 快速测试（推荐先用少量数据测试）

```bash
# 添加50张图像用于快速测试
python quick_setup.py 50
```

**预期输出**：
- 显示处理批次和成功添加的图像数量
- 最后显示数据库中共有50张图像

### 5.2 添加更多图像（测试通过后）

```bash
# 添加200张图像（用于充分测试）
python quick_setup.py 200

# 或添加更多（根据需求）
python quick_setup.py 500
python quick_setup.py 1000
```

---

## 步骤6：验证数据库

### 6.1 检查数据库状态

```bash
python -c "
from src.search.semantic_search import SemanticSearchEngine

engine = SemanticSearchEngine()
info = engine.get_system_info()
print('数据库图像数量:', info['database']['total_images'])
print('模型:', info['model']['name'])
print('设备:', info['model']['device'])
"
```

### 6.2 检查距离度量

```bash
python -c "
from src.database.chroma_db import ChromaDBManager
db = ChromaDBManager()
metadata = db.collection.metadata
print('Chroma元数据:', metadata)
# 应该显示 'hnsw:space': 'ip'
"
```

---

## 步骤7：启动服务

```bash
# 确保在text2img环境中
conda activate text2img

# 启动服务
python main.py
```

**预期输出**：
- INFO: Started server process [...]
- INFO: Waiting for application startup.
- 语义搜索引擎初始化成功
- INFO: Application startup complete.
- INFO: Uvicorn running on http://0.0.0.0:9000

---

## 步骤8：测试搜索功能

### 8.1 测试系统信息

```bash
# 新开一个终端窗口
python -c "
import requests
r = requests.get('http://localhost:9000/system/info')
print('系统信息:', r.json())
"
```

### 8.2 测试文本搜索

```bash
python -c "
import requests

# 测试多个搜索查询
queries = ['连衣裙', '红色衣服', 'T恤', '鞋子', '时尚']

for query in queries:
    response = requests.post(
        'http://localhost:9000/search/text',
        data={'query': query, 'top_k': 5},
        timeout=10
    )
    if response.status_code == 200:
        result = response.json()
        filtered = result.get('filtered_results', 0)
        raw_sim = result.get('raw_similarities', [])
        print(f'查询: {query}')
        print(f'  过滤后结果数: {filtered}')
        if raw_sim:
            print(f'  相似度分数: {[round(s, 3) for s in raw_sim[:3]]}')
        print()
"
```

### 8.3 使用测试脚本

```bash
# 运行完整测试脚本
python test_real_search.py
```

### 8.4 访问API文档

在浏览器中打开：
```
http://localhost:9000/docs
```

可以：
- 查看所有API接口
- 在线测试API
- 查看请求/响应格式

---

## 步骤9：验证修复效果

### 9.1 验证向量归一化

```bash
python -c "
from src.models.clip_model import CLIPModelWrapper
import numpy as np

model = CLIPModelWrapper()
test_image = 'data/raw/kaggle/fashion/fashion-dataset/images/10000.jpg'
test_text = '红色连衣裙'

img_emb = model.get_image_embedding(test_image)
text_emb = model.get_text_embedding(test_text)

img_norm = np.linalg.norm(img_emb)
text_norm = np.linalg.norm(text_emb)

print(f'图像向量范数: {img_norm:.6f}')
print(f'文本向量范数: {text_norm:.6f}')
print(f'归一化检查: {\"通过\" if abs(img_norm - 1.0) < 0.01 and abs(text_norm - 1.0) < 0.01 else \"失败\"}')
"
```

### 9.2 验证距离度量

```bash
python -c "
from src.database.chroma_db import ChromaDBManager
import yaml

# 检查配置
config = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
print('配置中的距离度量:', config['database']['chroma']['distance_metric'])

# 检查实际数据库
db = ChromaDBManager()
print('数据库中的距离度量:', db.collection.metadata.get('hnsw:space', 'N/A'))
"
```

### 9.3 测试点积距离计算

```bash
python -c "
from src.models.clip_model import CLIPModelWrapper
from src.database.chroma_db import ChromaDBManager
import numpy as np

# 获取查询向量（已归一化）
model = CLIPModelWrapper()
query_text = '红色连衣裙'
query_emb = model.get_text_embedding(query_text)[0]  # 取第一个

# 从数据库获取一个图像向量（已归一化）
db = ChromaDBManager()
if db.collection.count() > 0:
    # 获取第一个向量
    results = db.collection.get(limit=1)
    if results['embeddings']:
        img_emb = np.array(results['embeddings'][0])
        
        # 计算点积（归一化向量的点积 = 余弦相似度）
        dot_product = np.dot(query_emb, img_emb)
        print(f'点积值: {dot_product:.6f}')
        print(f'范围: [-1, 1] (归一化向量点积)')
        print(f'值越大表示越相似')
"
```

---

## 常见问题排查

### 问题1：服务启动失败

**检查**：
```bash
# 检查端口是否被占用
netstat -ano | findstr :9000

# 检查依赖是否安装
python -c "import fastapi, uvicorn, transformers, chromadb; print('依赖OK')"
```

### 问题2：搜索返回空结果

**可能原因**：
- 数据库为空
- 相似度阈值太高

**检查**：
```bash
# 检查数据库数量
python -c "from src.search.semantic_search import SemanticSearchEngine; e = SemanticSearchEngine(); print('图像数量:', e.get_system_info()['database']['total_images'])"

# 降低相似度阈值测试
# 编辑 config/config.yaml，将 similarity_threshold 改为 0.1
```

### 问题3：向量归一化不正确

**检查**：
```bash
python -c "
from src.models.clip_model import CLIPModelWrapper
import numpy as np

model = CLIPModelWrapper()
emb = model.get_image_embedding('data/raw/kaggle/fashion/fashion-dataset/images/10000.jpg')
norm = np.linalg.norm(emb)
if abs(norm - 1.0) > 0.01:
    print(f'警告: 向量范数 {norm:.6f} 不等于1.0')
else:
    print('向量归一化正常')
"
```

### 问题4：距离度量不匹配

**检查**：
```bash
# 确保配置和数据库一致
python -c "
import yaml
from src.database.chroma_db import ChromaDBManager

config = yaml.safe_load(open('config/config.yaml', 'r', encoding='utf-8'))
db = ChromaDBManager()

config_metric = config['database']['chroma']['distance_metric']
db_metric = db.collection.metadata.get('hnsw:space', 'N/A')

print(f'配置: {config_metric}, 数据库: {db_metric}')
if config_metric != db_metric:
    print('警告: 距离度量不匹配，需要重建数据库')
else:
    print('距离度量匹配')
"
```

---

## 完整测试流程

### 快速验证（5分钟）

```bash
# 1. 停止服务
# Ctrl+C 或 kill进程

# 2. 删除旧数据库
Remove-Item -Recurse -Force data\chroma_db

# 3. 激活环境
conda activate text2img

# 4. 添加50张测试
python quick_setup.py 50

# 5. 启动服务
python main.py

# 6. 测试（新终端）
python -c "import requests; r = requests.post('http://localhost:9000/search/text', data={'query': '衣服', 'top_k': 3}); print(r.json())"
```

### 完整测试（30分钟）

```bash
# 1-4 同上

# 5. 添加更多数据
python quick_setup.py 500

# 6. 运行完整测试
python test_real_search.py

# 7. 访问API文档测试
# 浏览器打开 http://localhost:9000/docs
```

---

## 验证清单

完成重建和测试后，确认以下项目：

- [ ] 旧数据库已删除
- [ ] 新数据库已创建（使用ip距离度量）
- [ ] 向量归一化正常（范数≈1.0）
- [ ] 服务正常启动
- [ ] 文本搜索返回结果
- [ ] 相似度分数合理（0-1之间）
- [ ] API文档可访问
- [ ] 系统信息接口正常

---

## 下一步

重建和测试完成后，您可以：

1. **添加更多数据**：使用 `python quick_setup.py 1000` 添加更多图像
2. **调整相似度阈值**：在 `config/config.yaml` 中调整 `similarity_threshold`
3. **性能测试**：测试大量并发请求
4. **准备训练数据**：如果有25万图文对，可以开始续训

---

## 注意事项

1. **数据备份**：重建前确保有原始图像数据备份
2. **环境一致**：确保使用text2img环境
3. **GPU可用**：如果使用CUDA，确保GPU正常
4. **磁盘空间**：25万图像向量数据库需要约几GB空间
5. **时间消耗**：添加大量图像需要时间（1000张约需10-20分钟）

---

## 快速参考命令

```bash
# 一键重建（50张测试）
conda activate text2img
Remove-Item -Recurse -Force data\chroma_db
python quick_setup.py 50
python main.py

# 测试
python -c "import requests; print(requests.get('http://localhost:9000/system/info').json())"
```

