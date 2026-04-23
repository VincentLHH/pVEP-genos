"""
tests/test_05_api_client.py
=============================
测试 API Client（无 GPU 节点，调用远程 GPU 服务）。

覆盖场景
--------
1. EmbeddingAPIClient 构造与 health 检查
2. bulk_get_embeddings 远程推理
3. health() / cache_size() / clear_cache() 工具方法
4. 与 EmbeddingManager 接口对齐（mock 结果验证）
5. 错误处理（服务不可用、超时）
6. 上下文管理器（__enter__ / __exit__）
7. 空输入 / dedup

前置条件
--------
- 远程 GPU 节点已启动 api.service（见 test_04）
- 本节点可以是任意机器（无 GPU 要求）

运行（无 GPU 节点）
-------------------
# 先在 GPU 节点启动服务（见 test_04）
# 然后在任意节点运行：
export PVEPGENOS_API_URL=http://gpu-node:8000
pytest tests/test_05_api_client.py -v
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.client import EmbeddingAPIClient, MultiEmbeddingAPIClient


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_base_url(env_api_url):
    return env_api_url


@pytest.fixture(scope="session")
def api_reachable(api_base_url, request):
    """确保 API 服务可访问；不可达时 skip"""
    try:
        client = EmbeddingAPIClient(base_url=api_base_url, timeout=5)
        health = client.health()
        return health
    except Exception as e:
        pytest.skip(f"API not reachable at {api_base_url}: {e}")


@pytest.fixture
def api_health(api_base_url):
    """基础连接检查：server 不可达则 skip"""
    import httpx
    try:
        httpx.get(f"{api_base_url}/health", timeout=5)
    except Exception as e:
        pytest.skip(f"API not reachable at {api_base_url}: {e}")


@pytest.fixture
def api_client(api_base_url):
    """每个测试使用独立的 API client；server 不可用时 skip"""
    import httpx
    try:
        httpx.get(f"{api_base_url}/health", timeout=5)
    except Exception as e:
        pytest.skip(f"API not reachable at {api_base_url}: {e}", allow_module_level=True)
    client = EmbeddingAPIClient(base_url=api_base_url, timeout=120)
    yield client
    client.close()


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def assert_client_result(result, methods):
    assert isinstance(result, dict)
    assert len(result) > 0
    for key, method_dict in result.items():
        assert isinstance(method_dict, dict)
        for method in methods:
            assert method in method_dict
            vec = method_dict[method]
            assert isinstance(vec, list)
            assert len(vec) >= 64
            assert not all(v == 0.0 for v in vec)


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestAPIClientConstruction:
    """构造与连接"""

    def test_health_check(self, api_reachable):
        """health() 应返回服务状态"""
        assert api_reachable["status"] == "ok"
        assert "model_name" in api_reachable
        print(f"API reachable: {api_reachable}")

    def test_model_name_property(self, api_client):
        """model_name property 代理 health()"""
        assert api_client.model_name == "Genos-1.2B"

    def test_context_manager(self, api_base_url, api_health):
        """支持 with 语法"""
        with EmbeddingAPIClient(base_url=api_base_url, timeout=30) as client:
            health = client.health()
            assert health["status"] == "ok"
        # __exit__ 会自动 close
        # 再次创建应成功
        client2 = EmbeddingAPIClient(base_url=api_base_url, timeout=30)
        assert client2.health()["status"] == "ok"
        client2.close()


class TestAPIClientBulkEmbed:
    """批量推理"""

    def test_bulk_get_embeddings_basic(self, api_client):
        """基础调用"""
        seqs = {
            "seq1": "ACGTACGTACGTACGT",
            "seq2": "TTTTCCCCGGGGAAAA",
        }
        result = api_client.bulk_get_embeddings(seqs, methods=["mean"])

        assert_client_result(result, ["mean"])
        print(f"bulk: {len(result)} seqs")

    def test_all_pooling_methods(self, api_client):
        """多 pooling 方法"""
        seqs = {"s1": "ACGT" * 16}
        result = api_client.bulk_get_embeddings(seqs, methods=["mean", "max", "last_token"])

        assert_client_result(result, ["mean", "max", "last_token"])

    def test_dedup_equivalent_to_api(self, api_client):
        """相同序列去重与 API 服务行为一致"""
        seqs = {
            "a": "ACGTACGTACGT",
            "b": "ACGTACGTACGT",  # same
        }
        result = api_client.bulk_get_embeddings(seqs, methods=["mean"])

        assert result["a"]["mean"] == result["b"]["mean"]

    def test_empty_input(self, api_client):
        """空输入返回空 dict"""
        result = api_client.bulk_get_embeddings({}, methods=["mean"])
        assert result == {}


class TestAPIClientCacheTools:
    """Cache 工具方法"""

    def test_cache_size(self, api_client):
        """cache_size() 返回整数"""
        size = api_client.cache_size()
        assert isinstance(size, int)
        assert size >= 0
        print(f"server cache size: {size}")

    def test_clear_cache(self, api_client):
        """clear_cache() 返回清空记录"""
        result = api_client.clear_cache()
        assert "cleared" in result
        assert isinstance(result["cleared"], int)
        print(f"cache cleared: {result}")


class TestAPIClientCompatibility:
    """与 EmbeddingManager 接口对齐"""

    def test_bulk_get_embeddings_signature(self, api_client):
        """签名与 EmbeddingManager.bulk_get_embeddings 一致"""
        import inspect
        sig = inspect.signature(api_client.bulk_get_embeddings)
        params = list(sig.parameters.keys())

        # 必须有 seq_dict 和 methods
        assert "seq_dict" in params
        assert "methods" in params

    def test_model_name_same_as_manager(self, api_client):
        """API client 的 model_name 与服务 health 一致"""
        assert api_client.model_name == api_client.health()["model_name"]

    def test_cache_property_empty_local(self, api_client):
        """API 模式下 cache property 返回空 dict（服务端管理）"""
        assert api_client.cache == {}


class TestAPIClientMulti:
    """MultiEmbeddingAPIClient"""

    def test_multi_client_creation(self, api_base_url):
        """MultiEmbeddingAPIClient 支持单 url 构造"""
        client = MultiEmbeddingAPIClient(base_url=api_base_url, timeout=30)
        assert client._default_url == api_base_url
        print("MultiEmbeddingAPIClient: created with single base_url")

    def test_multi_get_embeddings(self, api_base_url, api_health):
        """MultiEmbeddingAPIClient.bulk_get_embeddings 透传"""
        client = MultiEmbeddingAPIClient(base_url=api_base_url, timeout=30)
        result = client.get_embeddings({"s1": "ACGT" * 16}, methods=["mean"])

        assert isinstance(result, dict)
        assert "s1" in result
        client.close()


class TestAPIClientErrorHandling:
    """错误处理"""

    def test_wrong_port_timeout(self):
        """错误地址应抛异常"""
        client = EmbeddingAPIClient(base_url="http://localhost:99999", timeout=3)
        with pytest.raises(Exception):  # httpx 连接错误
            client.health()
        client.close()

    def test_invalid_url_format(self):
        """无效 URL 格式应抛出有意义的错误"""
        # 不应以 http:// 开头
        with pytest.raises(Exception):
            _ = EmbeddingAPIClient(base_url="invalid-url")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
