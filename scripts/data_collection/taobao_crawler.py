#!/usr/bin/env python3
"""
淘宝开放平台数据采集模块
用于获取商品图文数据构建训练数据集
"""

import os
import time
import json
import requests
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode
import hashlib
import hmac
import base64


class TaobaoAPIClient:
    """淘宝开放平台API客户端"""
    
    def __init__(self, app_key: str, app_secret: str, access_token: str = None):
        """
        初始化淘宝API客户端
        
        Args:
            app_key: 应用Key
            app_secret: 应用密钥
            access_token: 访问令牌（可选）
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = access_token
        self.base_url = "http://gw.api.taobao.com/router/rest"
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _generate_sign(self, params: Dict[str, Any]) -> str:
        """
        生成API签名
        
        Args:
            params: 请求参数
            
        Returns:
            签名字符串
        """
        # 按参数名排序
        sorted_params = sorted(params.items())
        
        # 拼接字符串
        string_to_sign = self.app_secret
        for key, value in sorted_params:
            if key != 'sign' and value is not None:
                string_to_sign += f"{key}{value}"
        string_to_sign += self.app_secret
        
        # 计算MD5
        sign = hashlib.md5(string_to_sign.encode('utf-8')).hexdigest().upper()
        return sign
    
    def call_api(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用淘宝API
        
        Args:
            method: API方法名
            params: API参数
            
        Returns:
            API响应数据
        """
        # 基础参数
        base_params = {
            'method': method,
            'app_key': self.app_key,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'format': 'json',
            'v': '2.0',
            'sign_method': 'md5'
        }
        
        # 添加访问令牌（如果有）
        if self.access_token:
            base_params['session'] = self.access_token
        
        # 合并参数
        all_params = {**base_params, **params}
        
        # 生成签名
        all_params['sign'] = self._generate_sign(all_params)
        
        try:
            # 发送请求
            response = requests.post(self.base_url, data=all_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 检查错误
            if 'error_response' in result:
                error = result['error_response']
                self.logger.error(f"API调用失败: {error.get('msg', 'Unknown error')}")
                return None
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return None
    
    def search_products(self, keyword: str, page_no: int = 1, page_size: int = 40) -> List[Dict[str, Any]]:
        """
        搜索商品
        
        Args:
            keyword: 搜索关键词
            page_no: 页码
            page_size: 每页数量
            
        Returns:
            商品列表
        """
        params = {
            'q': keyword,
            'page_no': page_no,
            'page_size': page_size,
            'fields': 'num_iid,title,pic_url,price,detail_url,seller_nick,volume'
        }
        
        result = self.call_api('taobao.items.search', params)
        
        if result and 'items_search_response' in result:
            items_data = result['items_search_response']
            if 'items' in items_data and 'item' in items_data['items']:
                return items_data['items']['item']
        
        return []
    
    def get_product_detail(self, num_iid: str) -> Optional[Dict[str, Any]]:
        """
        获取商品详情
        
        Args:
            num_iid: 商品ID
            
        Returns:
            商品详情
        """
        params = {
            'num_iid': num_iid,
            'fields': 'desc,item_img_urls,prop_img_urls,sku,props_name'
        }
        
        result = self.call_api('taobao.item.get', params)
        
        if result and 'item_get_response' in result:
            return result['item_get_response'].get('item', None)
        
        return None
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """
        下载商品图片
        
        Args:
            image_url: 图片URL
            save_path: 保存路径
            
        Returns:
            是否成功
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"图片下载成功: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"图片下载失败 {image_url}: {e}")
            return False


class TaobaoDataCollector:
    """淘宝数据收集器"""
    
    def __init__(self, app_key: str, app_secret: str, access_token: str = None):
        """
        初始化数据收集器
        
        Args:
            app_key: 应用Key
            app_secret: 应用密钥
            access_token: 访问令牌
        """
        self.api_client = TaobaoAPIClient(app_key, app_secret, access_token)
        self.logger = logging.getLogger(__name__)
    
    def collect_product_data(self, keywords: List[str], max_products_per_keyword: int = 1000,
                           output_dir: str = "data/raw/taobao") -> List[Dict[str, Any]]:
        """
        收集商品数据
        
        Args:
            keywords: 搜索关键词列表
            max_products_per_keyword: 每个关键词最大商品数量
            output_dir: 输出目录
            
        Returns:
            收集的商品数据列表
        """
        all_products = []
        
        for keyword in keywords:
            self.logger.info(f"开始收集关键词: {keyword}")
            
            products = self._collect_products_by_keyword(
                keyword, max_products_per_keyword, output_dir
            )
            
            all_products.extend(products)
            
            # 避免请求过于频繁
            time.sleep(2)
        
        # 保存数据
        self._save_dataset(all_products, output_dir)
        
        return all_products
    
    def _collect_products_by_keyword(self, keyword: str, max_products: int,
                                   output_dir: str) -> List[Dict[str, Any]]:
        """
        按关键词收集商品数据
        """
        products = []
        page_no = 1
        page_size = 40
        
        while len(products) < max_products:
            self.logger.info(f"搜索 {keyword} - 第 {page_no} 页")
            
            # 搜索商品
            search_results = self.api_client.search_products(keyword, page_no, page_size)
            
            if not search_results:
                self.logger.info(f"关键词 {keyword} 没有更多结果")
                break
            
            for product in search_results:
                if len(products) >= max_products:
                    break
                
                # 获取商品详情
                detail = self.api_client.get_product_detail(product['num_iid'])
                
                if detail:
                    # 构建数据记录
                    record = self._build_product_record(product, detail, output_dir)
                    
                    if record:
                        products.append(record)
                        self.logger.info(f"收集商品: {record['title'][:30]}...")
                
                # 避免请求过于频繁
                time.sleep(0.5)
            
            page_no += 1
            
            # 检查是否还有更多页面
            if len(search_results) < page_size:
                break
        
        return products
    
    def _build_product_record(self, product: Dict[str, Any], detail: Dict[str, Any],
                            output_dir: str) -> Optional[Dict[str, Any]]:
        """
        构建商品数据记录
        """
        try:
            # 下载主图
            image_filename = f"{product['num_iid']}_main.jpg"
            image_path = os.path.join(output_dir, "images", image_filename)
            
            if not self.api_client.download_image(product['pic_url'], image_path):
                return None
            
            # 构建记录
            record = {
                'product_id': product['num_iid'],
                'title': product['title'],
                'price': product.get('price', 0),
                'sales_volume': product.get('volume', 0),
                'seller': product.get('seller_nick', ''),
                'image_path': image_path,
                'image_url': product['pic_url'],
                'detail_url': product.get('detail_url', ''),
                'description': detail.get('desc', ''),
                'category': self._extract_category(detail),
                'attributes': self._extract_attributes(detail),
                'search_keyword': '',  # 将在外部设置
                'collection_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return record
            
        except Exception as e:
            self.logger.error(f"构建商品记录失败 {product.get('num_iid', 'unknown')}: {e}")
            return None
    
    def _extract_category(self, detail: Dict[str, Any]) -> str:
        """提取商品分类"""
        # 从商品属性中提取分类信息
        props_name = detail.get('props_name', '')
        # 这里可以添加更复杂的分类提取逻辑
        return props_name.split(':')[-1] if ':' in props_name else ''
    
    def _extract_attributes(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        """提取商品属性"""
        attributes = {}
        
        # 从商品详情中提取属性
        props_name = detail.get('props_name', '')
        if props_name:
            # 简单的属性解析逻辑
            for prop in props_name.split(';'):
                if ':' in prop:
                    key, value = prop.split(':', 1)
                    attributes[key.strip()] = value.strip()
        
        return attributes
    
    def _save_dataset(self, products: List[Dict[str, Any]], output_dir: str):
        """
        保存数据集
        """
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # 保存JSON数据
        dataset_path = os.path.join(output_dir, "taobao_dataset.json")
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"数据集已保存: {dataset_path}")
        self.logger.info(f"共收集 {len(products)} 个商品")


def main():
    """主函数 - 示例用法"""
    # 从环境变量获取认证信息
    app_key = os.getenv('TAOBAO_APP_KEY')
    app_secret = os.getenv('TAOBAO_APP_SECRET')
    access_token = os.getenv('TAOBAO_ACCESS_TOKEN')
    
    if not app_key or not app_secret:
        print("请设置环境变量: TAOBAO_APP_KEY, TAOBAO_APP_SECRET")
        return
    
    # 初始化收集器
    collector = TaobaoDataCollector(app_key, app_secret, access_token)
    
    # 定义搜索关键词
    keywords = [
        "连衣裙", "T恤", "牛仔裤", "运动鞋", "手机",
        "笔记本电脑", "化妆品", "家居用品", "食品"
    ]
    
    # 收集数据
    products = collector.collect_product_data(
        keywords=keywords,
        max_products_per_keyword=100,  # 每个关键词收集100个商品
        output_dir="data/raw/taobao"
    )
    
    print(f"数据收集完成，共收集 {len(products)} 个商品")


if __name__ == "__main__":
    main()