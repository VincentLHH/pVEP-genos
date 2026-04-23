"""
api/service.py
===============
Embedding 推理 API 服务（供 GPU 节点部署使用）。

启动方式
--------
    python -m api.service --model-name Genos-1.2B \\
        --model-path /path/to/model \\
        --device cuda:0 \\
        --port 8000

API 端点
--------
    GET  /health              健康检查 + cache 统计
    POST /embed               批量推理（与 EmbeddingManager.bulk_get_embeddings 接口对齐）
    GET  /cache/size          返回 cache 当前条目数
    DELETE /cache             清空 cache（管理员用）
"""

import argparse
import sys
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.embedding_manager import EmbeddingManager


# =========================================================
# 全局状态（单例，服务生命周期内有效）
# =========================================================
_manager: Optional[EmbeddingManager] = None
_startup_lock = threading.Lock()


# =========================================================
# Pydantic 请求/响应模型
# =========================================================
class EmbedRequest(BaseModel):
    seq_dict: Dict[str, str] = Field(
        ...,
        description='映射关系：{flat_key: sequence_str}',
        examples=[{"var1||hap1_mut_seq": "ACGTACGT..."}]
    )
    methods: List[str] = Field(
        default=["mean"],
        description='Pooling 方法列表，支持 mean / max / last_token'
    )


class EmbedItem(BaseModel):
    mean: Optional[List[float]] = None
    max: Optional[List[float]] = None
    last_token: Optional[List[float]] = None


class EmbedResponse(BaseModel):
    result: Dict[str, Dict[str, List[float]]]
    cache_hits: int = Field(description="本次请求中 cache 命中的 key 数")
    total_cache_size: int = Field(description="服务全局 cache 当前条目数")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    cache_size: int
    version: str = "1.0"


# =========================================================
# FastAPI app
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期管理。"""
    # _args_holder 在 run_server() 里已填充，在 uvicorn worker 进程中均可访问
    host = _args_holder.get("host", "0.0.0.0")
    port = _args_holder.get("port", 8000)
    print(f"[API Service] ✅ Ready on {host}:{port}")
    yield
    print("[API Service] 🔻 Shutting down...")


app = FastAPI(
    title="pVEP-genos Embedding API",
    description="Genos 模型批量 embedding 推理服务，支持 cache 复用。",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """健康检查与 cache 状态。"""
    if _manager is None:
        raise HTTPException(503, "Model not loaded yet")
    return HealthResponse(
        status="ok",
        model_name=_manager.model_name,
        device=_manager.device,
        cache_size=len(_manager.cache),
    )


@app.post("/embed", response_model=EmbedResponse, tags=["Inference"])
async def embed(req: EmbedRequest):
    """
    批量推理接口。

    - seq_dict : {key: sequence} 扁平映射
    - methods  : ["mean", "max", "last_token"] 的子集

    返回
    ----
    result       : {key: {method: [float, ...]}}
    cache_hits   : 本次请求中从 cache 直接返回的 key 数
    total_cache_size: 服务全局 cache 当前大小
    """
    if _manager is None:
        raise HTTPException(503, "Model not loaded yet")

    if not req.seq_dict:
        return EmbedResponse(result={}, cache_hits=0, total_cache_size=len(_manager.cache))

    # 预热阶段：先记录哪些 key 在推理前就已经命中 cache
    cache_hit_keys = [
        key for key, seq in req.seq_dict.items()
        if all((seq, m) in _manager.cache for m in req.methods)
    ]

    result = _manager.bulk_get_embeddings(req.seq_dict, req.methods)

    return EmbedResponse(
        result=result,
        cache_hits=len(cache_hit_keys),
        total_cache_size=len(_manager.cache),
    )


@app.get("/cache/size", tags=["Cache"])
async def cache_size():
    """返回当前 cache 条目数。"""
    if _manager is None:
        return {"size": 0}
    return {"size": len(_manager.cache)}


@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """清空全局 cache（慎用）。"""
    if _manager is None:
        raise HTTPException(503, "Model not loaded yet")
    size = len(_manager.cache)
    _manager.cache.clear()
    return {"cleared": size, "message": f"Cleared {size} entries"}


# =========================================================
# CLI 启动入口
# =========================================================
# 用 module-level 全局存储解析后的 args，供 lifespan 闭包使用
_args_holder: Dict = {}


def run_server(
    model_name: str,
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 32,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    启动 API 服务（非 async，供 CLI 调用）。

    注意：固定单 worker（workers=1），因为 _manager 加载在主进程，
    若用 workers>1 会导致 worker 进程重新导入模块时 _manager=None。
    """
    global _manager, _args_holder

    print(f"[API Service] 🔄 Loading model [{model_name}] on {device}...")

    # 加载模型（同步，在主进程）
    _manager = EmbeddingManager(
        model_name=model_name,
        model_path=model_path,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        shared_cache=None,   # API 服务是单进程，不需要 Manager.dict()
    )

    _args_holder.update({"host": host, "port": port})

    # 直接传 app 对象（不传字符串），避免 uvicorn 重新导入模块导致 _manager 丢失
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(description="pVEP-genos Embedding API Server")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    global args
    args = parser.parse_args()

    run_server(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
