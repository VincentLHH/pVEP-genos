"""
tests/test_04_api_service.py
=============================
测试 API Service（GPU 节点 standalone 模式）。

覆盖场景
--------
1. API 服务启动 / 关闭生命周期
2. GET  /health 健康检查
3. POST /embed 批量推理
4. GET  /cache/size cache 统计
5. DELETE /cache 清空 cache
6. 空输入 / 异常输入容错
7. Cache 命中率统计

前置条件
--------
- GPU 节点（有 CUDA）
- API 服务可独立启动（不需要跑 run_pipeline.py）

运行（GPU 节点，两个终端）
----------------------------
# 终端 1：启动服务
export PVEPGENOS_MODEL_PATH=/path/to/Genos-1.2B
python -m api.service \
    --model-name Genos-1.2B \
    --model-path "$PVEPGENOS_MODEL_PATH" \
    --device cuda:0 \
    --port 8000

# 终端 2：运行测试
pytest tests/test_04_api_service.py -v

标记过滤：
pytest tests/ -v -m "api"          # 仅运行 API 测试
pytest tests/ -v -m "not api"       # 跳过 API 测试
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# 使用 conftest.py 统一的环境检测和 skip helpers
# ─────────────────────────────────────────────────────────────
from tests.conftest import (
    HAS_CUDA,
    skip_if_no_cuda,
    require_api_service,
)

# ─────────────────────────────────────────────────────────────
# Fixtures（复用 conftest 的通用 fixtures）
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_health(require_api_service):
    """确保 API 服务在线（conftest 提供）"""
    return require_api_service


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def assert_health_response(data):
    assert isinstance(data, dict)
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "device" in data
    assert "cache_size" in data


def assert_embed_response(data, n_keys, methods):
    assert isinstance(data, dict)
    assert "result" in data
    assert isinstance(data["result"], dict)
    assert set(data["result"].keys()) == set(n_keys) if isinstance(n_keys, list) else n_keys
    for key, methods_dict in data["result"].items():
        for method in methods:
            assert method in methods_dict
            vec = methods_dict[method]
            assert isinstance(vec, list)
            assert len(vec) >= 64
    assert "cache_hits" in data
    assert "total_cache_size" in data


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestAPIHealth:
    """健康检查（api 标记）"""

    @pytest.mark.api
    def test_health_ok(self, api_health):
        """服务正常时应返回 status=ok"""
        assert_health_response(api_health)
        print(f"API health: {api_health}")

    @pytest.mark.api
    def test_health_shows_model(self, api_health):
        """health 应显示模型名称"""
        assert api_health["model_name"] == "Genos-1.2B"

    @pytest.mark.api
    def test_health_shows_device(self, api_health):
        """health 应显示设备"""
        assert "cuda" in api_health["device"].lower() or "cpu" in api_health["device"].lower()


class TestAPICache:
    """Cache 管理（api 标记）"""

    @pytest.mark.api
    def test_cache_size(self, require_api_service):
        """GET /cache/size 应返回 cache 条目数"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")
        resp = httpx.get(f"{base_url}/cache/size", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "size" in data
        assert isinstance(data["size"], int)
        print(f"cache size: {data['size']}")

    @pytest.mark.api
    def test_clear_cache(self, require_api_service):
        """DELETE /cache 应清空 cache"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        # 先清空
        resp = httpx.delete(f"{base_url}/cache", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "cleared" in data
        print(f"cache cleared: {data['cleared']} entries removed")

        # 验证 size 为 0
        resp2 = httpx.get(f"{base_url}/cache/size", timeout=10)
        resp2.raise_for_status()
        assert resp2.json()["size"] == 0


class TestAPIEmbed:
    """批量推理（api 标记）"""

    @pytest.mark.api
    def test_embed_basic(self, require_api_service):
        """POST /embed 基本调用"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        payload = {
            "seq_dict": {
                "seq1": "ACGTACGTACGTACGT",
                "seq2": "TTTTCCCCGGGGAAAA",
            },
            "methods": ["mean"],
        }
        resp = httpx.post(f"{base_url}/embed", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        assert_embed_response(data, ["seq1", "seq2"], ["mean"])
        print(f"embed: {len(data['result'])} seqs, cache_hits={data['cache_hits']}")

    @pytest.mark.api
    def test_embed_returns_all_methods(self, require_api_service):
        """返回所有请求的 pooling 方法"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        payload = {
            "seq_dict": {"s1": "ACGT" * 20},
            "methods": ["mean", "max", "last_token"],
        }
        resp = httpx.post(f"{base_url}/embed", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        assert_embed_response(data, ["s1"], ["mean", "max", "last_token"])
        print(f"multi-method: mean/max/last_token all present")

    @pytest.mark.api
    def test_embed_dedup_on_server(self, require_api_service):
        """相同序列应在服务端被去重（cache 命中）"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        payload = {
            "seq_dict": {
                "a": "ACGTACGTACGT",
                "b": "ACGTACGTACGT",  # duplicate
            },
            "methods": ["mean"],
        }
        resp = httpx.post(f"{base_url}/embed", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # cache_hits 应包含 "a"（首次推理后）或两者都是 new
        # 关键是结果应该一致
        assert data["result"]["a"]["mean"] == data["result"]["b"]["mean"]
        print(f"dedup: a==b confirmed, cache_size={data['total_cache_size']}")

    @pytest.mark.api
    def test_empty_seq_dict(self, require_api_service):
        """空 seq_dict 应返回空 result"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        payload = {"seq_dict": {}, "methods": ["mean"]}
        resp = httpx.post(f"{base_url}/embed", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert data["result"] == {}

    @pytest.mark.api
    def test_cache_grows_after_requests(self, require_api_service):
        """连续请求后 cache 应持续增长"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        # 清空 cache
        httpx.delete(f"{base_url}/cache", timeout=10).raise_for_status()

        # 推理 5 条新序列（确保互不相同，避免被服务端 dedup 去重）
        seqs = {f"s{i}": f"ACGT" * 16 + chr(65 + i) * 1 for i in range(5)}
        payload = {"seq_dict": seqs, "methods": ["mean"]}
        resp = httpx.post(f"{base_url}/embed", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 全部是 new（cache_hits 应为 0）
        assert data["cache_hits"] == 0
        assert data["total_cache_size"] >= 5
        print(f"cache grew to {data['total_cache_size']} entries after 5 new seqs")


class TestAPIConcurrency:
    """并发请求（api + gpu 标记）"""

    @pytest.mark.api
    @pytest.mark.gpu
    @skip_if_no_cuda
    def test_concurrent_requests(self, require_api_service):
        """多个并发请求均应成功"""
        import httpx
        base_url = os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")

        def call_embed(seq_id):
            payload = {
                "seq_dict": {f"{seq_id}": "ACGT" * 32},
                "methods": ["mean"],
            }
            resp = httpx.post(f"{base_url}/embed", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(call_embed, i) for i in range(4)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 4
        for r in results:
            assert "result" in r
        print(f"4 concurrent requests: all succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
