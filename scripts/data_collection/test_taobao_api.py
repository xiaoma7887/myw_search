#!/usr/bin/env python3
"""
测试淘宝API连接
"""

import os
import sys
# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taobao_crawler import TaobaoAPIClient

def test_api_connection():
    """测试API连接"""
    
    # 从环境变量获取认证信息
    app_key = os.getenv('TAOBAO_APP_KEY')
    app_secret = os.getenv('TAOBAO_APP_SECRET')
    
    if not app_key or not app_secret:
        print("❌ 请先设置环境变量:")
        print("   export TAOBAO_APP_KEY=您的App Key")
        print("   export TAOBAO_APP_SECRET=您的App Secret")
        return False
    
    print("🔑 认证信息检查:")
    print(f"   App Key: {app_key[:10]}...")
    print(f"   App Secret: {app_secret[:10]}...")
    
    # 初始化API客户端
    try:
        client = TaobaoAPIClient(app_key, app_secret)
        print("✅ API客户端初始化成功")
        
        # 测试搜索功能
        print("🔍 测试商品搜索...")
        results = client.search_products("连衣裙", page_size=5)
        
        if results:
            print(f"✅ 搜索测试成功，找到 {len(results)} 个商品")
            for i, product in enumerate(results[:3]):
                print(f"   {i+1}. {product.get('title', 'N/A')}")
            return True
        else:
            print("❌ 搜索测试失败，请检查API权限")
            return False
            
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()