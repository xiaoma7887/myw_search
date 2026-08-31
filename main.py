#!/usr/bin/env python3
"""
语义商品搜索系统主程序
提供基于CLIP和Chroma的语义搜索服务
"""

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import shutil
from typing import List, Optional

from src.search.semantic_search import SemanticSearchEngine

# 全局搜索引擎实例
search_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化搜索引擎
    global search_engine
    try:
        search_engine = SemanticSearchEngine()
        print("语义搜索引擎初始化成功")
    except Exception as e:
        print(f"搜索引擎初始化失败: {e}")
        raise
    yield
    # 关闭时的清理工作（如果需要）


app = FastAPI(
    title="语义商品搜索系统",
    description="基于CLIP和Chroma的语义商品搜索API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根端点，返回服务状态"""
    return {
        "service": "语义商品搜索系统",
        "status": "运行中",
        "version": "1.0.0"
    }


@app.get("/system/info")
async def get_system_info():
    """获取系统信息"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    try:
        info = search_engine.get_system_info()
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")


@app.post("/search/text")
async def search_by_text(
    query: str = Form(..., description="搜索查询文本"),
    top_k: Optional[int] = Form(10, description="返回结果数量")
):
    """
    根据文本查询搜索相似商品
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    try:
        results = search_engine.search_by_text(query, top_k)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.post("/search/image")
async def search_by_image(
    image: UploadFile = File(..., description="查询图像文件"),
    top_k: Optional[int] = Form(10, description="返回结果数量")
):
    """
    根据图像搜索相似商品
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    # 验证文件类型
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图像格式")
    
    try:
        # 保存上传的图像到临时文件
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, image.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 执行搜索
        results = search_engine.search_by_image(temp_path, top_k)
        
        # 清理临时文件
        os.remove(temp_path)
        
        return JSONResponse(content=results)
    except Exception as e:
        # 确保清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"图像搜索失败: {str(e)}")


@app.post("/index/add")
async def add_to_index(
    images: List[UploadFile] = File(..., description="要添加的图像文件列表"),
    metadata: Optional[str] = Form(None, description="元数据JSON字符串")
):
    """
    添加图像到搜索索引
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    if not images:
        raise HTTPException(status_code=400, detail="至少需要上传一张图像")
    
    try:
        # 保存上传的图像到临时目录
        temp_dir = "temp_upload"
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        for image in images:
            if not image.content_type.startswith('image/'):
                continue
            
            temp_path = os.path.join(temp_dir, image.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            image_paths.append(temp_path)
        
        # 解析元数据
        metadatas = None
        if metadata:
            import json
            metadatas = json.loads(metadata)
        
        # 添加到索引
        result = search_engine.add_images_to_index(image_paths, metadatas)
        
        # 清理临时文件
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return JSONResponse(content=result)
    except Exception as e:
        # 清理临时文件
        if 'image_paths' in locals():
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
        raise HTTPException(status_code=500, detail=f"添加索引失败: {str(e)}")


@app.delete("/index/reset")
async def reset_index():
    """重置搜索索引"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    try:
        result = search_engine.reset_index()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置索引失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if not search_engine:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "搜索引擎未初始化"}
        )
    
    try:
        # 简单的健康检查：尝试获取系统信息
        search_engine.get_system_info()
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": str(e)}
        )


if __name__ == "__main__":
    # 加载配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    server_config = config['server']
    
    # 启动服务
    uvicorn.run(
        app,
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['debug']
    )