"""
api/client.py
=============
Embedding API 客户端（供无 GPU 节点调用远程推理服务）。

使用方式
--------
    client = EmbeddingAPIClient(
        base_url="http://gpu-node:8000",
        timeout=300,
    )
    result = client.bulk_get_embeddings({"key1": "ACGT...", "key2": "TGCA..."}, ["mean"])
"""

import os
from typing import Dict, List, Optional

import httpx

class EmbeddingAPIClient:
    """
    HTTP API 客户端，复刻 EmbeddingManager.bulk_get_embeddings 接口。

    参数
    ----
    base_url   : API 服务根地址，例如 http://192.168.1.100:8000
    timeout    : 请求超时（秒），默认 300s（大批量推理可能较慢）
    headers    : 可选额外 HTTP headers（如鉴权 token）
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._headers = headers or {}

    # --------------------------------------------------------------
    # 内部 httpx 客户端（延迟初始化，支持 __enter__ / __exit__ 上下文管理）
    # --------------------------------------------------------------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._headers,
            )
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --------------------------------------------------------------
    # API 查询
    # --------------------------------------------------------------
    def health(self) -> dict:
        """健康检查。"""
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def cache_size(self) -> int:
        """返回服务端 cache 当前条目数。"""
        resp = self.client.get("/cache/size")
        resp.raise_for_status()
        return resp.json()["size"]

    def clear_cache(self) -> dict:
        """清空服务端 cache。"""
        resp = self.client.delete("/cache")
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------
    # 🔥 核心接口：与 EmbeddingManager.bulk_get_embeddings 完全对齐
    # --------------------------------------------------------------
    def bulk_get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        批量推理接口。

        输入/输出格式与 EmbeddingManager.bulk_get_embeddings 完全一致，
        调用方无需感知是本地推理还是远程 API。

        参数
        ----
        seq_dict : {flat_key: sequence_str}
        methods  : pooling 方法列表

        返回
        ----
        {flat_key: {method: [float, ...]}}
        """
        if not seq_dict:
            return {}

        payload = {
            "seq_dict": seq_dict,
            "methods": methods,
        }

        resp = self.client.post("/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return data["result"]

    # --------------------------------------------------------------
    # 兼容包装：假装自己是 EmbeddingManager（仅 bulk_get_embeddings）
    # --------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.health()["model_name"]

    @property
    def cache(self) -> dict:
        """返回本地空 dict（API 模式的 cache 在服务端）。"""
        return {}


# =========================================================
# 多模型 API 客户端（可选，对齐 MultiEmbeddingManager）
# =========================================================
class MultiEmbeddingAPIClient:
    """
    多模型 API 客户端，对齐 MultiEmbeddingManager 接口。

    参数
    ----
    base_urls : {model_name: url} 映射，或单个 url（所有模型共用）
    timeout   : 请求超时（秒）
    """

    def __init__(
        self,
        base_urls: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
    ):
        if base_urls is None and base_url is None:
            raise ValueError("必须提供 base_urls 或 base_url")
        if base_urls is None:
            base_urls = {}

        self._clients: Dict[str, EmbeddingAPIClient] = {}
        self._default_url = base_url

        for name, url in base_urls.items():
            self._clients[name] = EmbeddingAPIClient(base_url=url, timeout=timeout)

    def _client_for(self, model_name: str) -> EmbeddingAPIClient:
        if model_name in self._clients:
            return self._clients[model_name]
        if self._default_url:
            return EmbeddingAPIClient(base_url=self._default_url, timeout=300.0)
        raise KeyError(f"No client configured for model: {model_name}")

    def get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, List[float]]]:
        """透传给单模型客户端（与 MultiEmbeddingManager 行为一致）。"""
        client = self._client_for(next(iter(self._clients)) if self._clients else "")
        return client.bulk_get_embeddings(seq_dict, methods)
