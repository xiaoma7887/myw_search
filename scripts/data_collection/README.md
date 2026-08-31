# 淘宝数据采集模块

本模块用于从淘宝开放平台获取商品图文数据，构建CLIP模型训练数据集。

## 🚀 快速开始

### 1. 申请淘宝开放平台权限

1. 访问 [淘宝开放平台](https://open.taobao.com/)
2. 注册开发者账号
3. 创建应用，获取以下信息：
   - `App Key`
   - `App Secret`
   - `Access Token`（可选）

### 2. 配置环境变量

```bash
# 设置淘宝API认证信息
export TAOBAO_APP_KEY="your_app_key"
export TAOBAO_APP_SECRET="your_app_secret"
export TAOBAO_ACCESS_TOKEN="your_access_token"  # 可选
```

### 3. 运行数据采集

```bash
# 直接运行采集脚本
python scripts/data_collection/taobao_crawler.py

# 或者使用配置化的方式
python scripts/data_collection/dataset_builder.py
```

## 📁 文件结构

```
scripts/data_collection/
├── __init__.py
├── README.md
├── taobao_crawler.py      # 淘宝API数据采集
└── dataset_builder.py     # 数据集构建工具
```

## 🔧 配置说明

### 淘宝API配置 (`config/taobao_config.yaml`)

```yaml
taobao:
  app_key: "${TAOBAO_APP_KEY}"
  app_secret: "${TAOBAO_APP_SECRET}"
  access_token: "${TAOBAO_ACCESS_TOKEN}"
  
  data_collection:
    keywords:
      - "连衣裙"
      - "T恤"
      - "牛仔裤"
    max_products_per_keyword: 200
    page_size: 40
    request_interval: 0.5
```

### 数据集配置

```yaml
dataset:
  split_ratio:
    train: 0.8
    val: 0.1
    test: 0.1
  
  preprocessing:
    text:
      min_length: 5
      max_length: 200
      remove_special_chars: true
    image:
      min_width: 100
      min_height: 100
```

## 📊 数据格式

### 原始数据格式
```json
{
  "product_id": "123456789",
  "title": "夏季新款白色蕾丝连衣裙",
  "price": 299.0,
  "image_path": "data/raw/taobao/images/123456789_main.jpg",
  "description": "商品详细描述...",
  "category": "女装/连衣裙",
  "attributes": {
    "颜色": "白色",
    "风格": "蕾丝"
  }
}
```

### 训练数据格式
```json
{
  "image_path": "data/raw/taobao/images/123456789_main.jpg",
  "text": "夏季新款白色蕾丝连衣裙",
  "product_id": "123456789",
  "category": "女装/连衣裙",
  "source": "taobao"
}
```

## ⚠️ 注意事项

1. **API限制**：遵守淘宝开放平台的调用频率限制
2. **数据合规**：仅用于研究和学习目的
3. **请求间隔**：设置合理的请求间隔避免被封禁
4. **数据质量**：定期检查采集的数据质量

## 🔄 工作流程

1. **数据采集** → 通过淘宝API获取商品信息
2. **图片下载** → 下载商品主图
3. **数据清洗** → 过滤无效数据和重复数据
4. **数据集构建** → 生成训练/验证/测试集
5. **质量检查** → 验证数据质量和一致性

## 📈 扩展建议

- 支持多个电商平台数据源
- 添加数据增强功能
- 实现增量数据采集
- 添加数据质量监控