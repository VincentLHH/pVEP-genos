"""
tests/test_03_embedding_manager.py
====================================
测试 EmbeddingManager 本地推理（需要 GPU 节点）。

覆盖场景
--------
1. 模型加载（CUDA / CPU fallback）
2. bulk_get_embeddings 基础调用
3. 序列去重（相同序列只推理一次，cache 命中）
4. 多种 pooling 方法（mean / max / last_token）
5. Batch 分块处理（大批量序列）
6. 空输入 / 单序列 / 多方法并发
7. shared_cache 注入（多卡场景的前置条件）
8. 兼容旧接口 get_embeddings()

前置条件
--------
- GPU 节点（有 CUDA + 模型权重）
- 环境变量 PVEPGENOS_MODEL_PATH 或 config/default.yaml 中有模型路径

运行（GPU 节点）
----------------
# 指定模型路径
export PVEPGENOS_MODEL_PATH=/path/to/Genos-1.2B
pytest tests/test_03_embedding_manager.py -v

# 或使用 CPU（慢，仅验证接口）
pytest tests/test_03_embedding_manager.py -v -k "cpu"
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# 环境检测
# ─────────────────────────────────────────────────────────────
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
except Exception:
    HAS_CUDA = False
    GPU_COUNT = 0

skip_if_no_cuda = pytest.mark.skipif(
    not HAS_CUDA,
    reason="CUDA not available; run on GPU node or use API mode"
)

skip_if_no_model = pytest.mark.skipif(
    True,  # 动态检测，见下面 fixture
    reason="Model path not configured"
)

from models.embedding_manager import EmbeddingManager


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_path(env_model_path):
    """优先从环境变量 / config 读取模型路径"""
    if env_model_path and os.path.exists(env_model_path):
        return env_model_path
    pytest.skip(f"Model path not found: {env_model_path}", allow_module_level=True)


@pytest.fixture(scope="session")
def manager_cuda(model_path):
    """本地 CUDA 推理 manager"""
    mgr = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path=model_path,
        device="cuda",
        dtype="bfloat16",
        batch_size=8,
        mode="local",
    )
    return mgr


@pytest.fixture(scope="session")
def manager_cpu(model_path):
    """CPU fallback manager（用于无 GPU 时验证接口）"""
    mgr = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path=model_path,
        device="cpu",
        dtype="float32",
        batch_size=2,
        mode="local",
    )
    return mgr


@pytest.fixture
def shared_cache():
    """干净的 shared cache dict（模拟多卡 Manager().dict()）"""
    return {}


# ─────────────────────────────────────────────────────────────
# 辅助断言
# ─────────────────────────────────────────────────────────────

def assert_embedding_result(result, expected_keys=None, min_dim=64):
    assert isinstance(result, dict), "result should be dict"
    assert len(result) > 0, "result should not be empty"

    for key, methods in result.items():
        assert isinstance(methods, dict), f"{key}: methods should be dict"
        if expected_keys:
            assert set(methods.keys()) == set(expected_keys), (
                f"{key}: expected methods {expected_keys}, got {methods.keys()}"
            )
        for method, vec in methods.items():
            assert isinstance(vec, list), f"{key}/{method}: should be list"
            assert len(vec) >= min_dim, f"{key}/{method}: dim={len(vec)} < {min_dim}"
            assert not all(v == 0.0 for v in vec), f"{key}/{method}: all zeros"


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestEmbeddingManagerLoad:
    """模型加载"""

    @skip_if_no_cuda
    def test_model_loads_on_cuda(self, model_path):
        """CUDA 模式下模型成功加载"""
        mgr = EmbeddingManager(
            model_name="TestModel",
            model_path=model_path,
            device="cuda",
            mode="local",
        )
        assert mgr.model is not None
        assert "cuda" in mgr.device
        print(f"model loaded on {mgr.device}, param dtype={next(mgr.model.parameters()).dtype}")

    def test_model_loads_on_cpu(self, model_path):
        """CPU 模式下模型成功加载（无 GPU 时 fallback）"""
        mgr = EmbeddingManager(
            model_name="TestModel",
            model_path=model_path,
            device="cpu",
            dtype="float32",
            mode="local",
        )
        assert mgr.model is not None
        assert mgr.device == "cpu"


class TestBulkGetEmbeddingsBasic:
    """基础推理功能"""

    @skip_if_no_cuda
    def test_basic_embedding_cuda(self, manager_cuda):
        """CUDA：单序列 embedding"""
        seqs = {"seq1": "ACGTACGTACGTACGT"}
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])
        assert_embedding_result(result, expected_keys=["mean"])

    def test_basic_embedding_cpu(self, manager_cpu):
        """CPU：单序列 embedding（无 GPU 时 fallback）"""
        seqs = {"seq1": "ACGTACGTACGTACGT"}
        result = manager_cpu.bulk_get_embeddings(seqs, methods=["mean"])
        assert_embedding_result(result, expected_keys=["mean"])

    @skip_if_no_cuda
    def test_multiple_sequences(self, manager_cuda):
        """CUDA：多序列同时推理"""
        seqs = {
            f"seq{i}": "ACGT" * 20 for i in range(10)
        }
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])
        assert_embedding_result(result, expected_keys=["mean"])
        assert len(result) == 10

    def test_empty_input(self, manager_cuda):
        """空输入返回空 dict"""
        result = manager_cuda.bulk_get_embeddings({}, methods=["mean"])
        assert result == {}

    def test_single_key_multiple_methods(self, manager_cuda):
        """单序列，多 pooling 方法"""
        seqs = {"seq1": "ACGTACGT" * 10}
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean", "max", "last_token"])
        assert_embedding_result(result, expected_keys=["mean", "max", "last_token"])


class TestSequenceDeduplication:
    """序列去重 + cache 命中"""

    @skip_if_no_cuda
    def test_duplicate_sequence_same_result(self, manager_cuda):
        """相同序列的去重 key 应得到相同 embedding"""
        seqs = {
            "key_a": "ACGTACGTACGT",
            "key_b": "ACGTACGTACGT",   # 与 key_a 完全相同
            "key_c": "TTTTCCCCGGGG",   # 不同
        }
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])

        assert result["key_a"]["mean"] == result["key_b"]["mean"], \
            "Duplicate sequences should have identical embeddings"
        assert result["key_a"]["mean"] != result["key_c"]["mean"], \
            "Different sequences should have different embeddings"

    @skip_if_no_cuda
    def test_cache_hit_on_second_call(self, manager_cuda):
        """第二次调用相同序列应直接命中 cache"""
        seqs = {"key1": "ACGT" * 20}
        # 第一次调用
        r1 = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])
        cache_size_before = len(manager_cuda.cache)

        # 第二次调用（全部命中）
        r2 = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])
        cache_size_after = len(manager_cuda.cache)

        assert r1 == r2, "Second call should return identical results"
        assert cache_size_before == cache_size_after, "Cache size should not grow on hit"
        print(f"cache size: before={cache_size_before}, after={cache_size_after} (hit confirmed)")


class TestBatchChunking:
    """Batch 分块（大序列量）"""

    @skip_if_no_cuda
    def test_large_batch_chunks(self, manager_cuda):
        """大批量序列（>batch_size）应正确分块"""
        # batch_size=8，构造 25 条序列
        seqs = {f"seq{i}": f"{'ACGT'[i%4]}" * 64 for i in range(25)}
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])

        assert len(result) == 25, f"Expected 25 results, got {len(result)}"
        assert_embedding_result(result, expected_keys=["mean"])
        print(f"large batch (25 seqs, batch_size=8): all {len(result)} returned")


class TestSharedCacheInjection:
    """shared_cache 注入（多卡 Manager().dict() 前置条件）"""

    @skip_if_no_cuda
    def test_shared_cache_populated(self, model_path):
        """通过 shared_cache 参数注入 dict，推理后 cache 内容存在其中"""
        shared = {}
        mgr = EmbeddingManager(
            model_name="TestShared",
            model_path=model_path,
            device="cuda",
            mode="local",
            shared_cache=shared,
        )

        seqs = {"s1": "ACGT" * 20}
        mgr.bulk_get_embeddings(seqs, methods=["mean"])

        # shared cache 应被填充
        assert len(shared) > 0, "shared_cache should be populated after inference"
        print(f"shared_cache entries: {len(shared)}")

    @skip_if_no_cuda
    def test_shared_cache_externally_filled(self, model_path):
        """外部预先填充 shared_cache，推理时应跳过已缓存的 key"""
        shared = {}
        # 预先写入一条 cache（模拟第一个 GPU 进程的结果）
        shared[("PRE_FILLED_SEQ", "mean")] = [0.123] * 128

        mgr = EmbeddingManager(
            model_name="TestPreFill",
            model_path=model_path,
            device="cuda",
            mode="local",
            shared_cache=shared,
        )

        seqs = {"pre": "PRE_FILLED_SEQ", "new": "ACGT" * 20}
        result = mgr.bulk_get_embeddings(seqs, methods=["mean"])

        # pre 应返回预先写入的值（而非重新推理）
        assert len(result["pre"]["mean"]) == 128
        # new 应有新结果
        assert len(result["new"]["mean"]) >= 64
        print("shared_cache pre-fill + new inference: PASS")


class TestLegacyInterface:
    """兼容旧接口 get_embeddings()"""

    @skip_if_no_cuda
    def test_get_embeddings_compat(self, manager_cuda):
        """get_embeddings() 应与 bulk_get_embeddings() 等价"""
        seqs = {"k1": "ACGT" * 10}

        r1 = manager_cuda.get_embeddings(seqs, methods=["mean"])
        r2 = manager_cuda.bulk_get_embeddings(seqs, methods=["mean"])

        assert r1 == r2, "get_embeddings() should behave identically to bulk_get_embeddings()"


class TestPoolingMethods:
    """多种 pooling 方法"""

    @skip_if_no_cuda
    def test_all_pooling_methods(self, manager_cuda):
        """mean / max / last_token 均应返回非零向量"""
        seqs = {"s1": "ACGT" * 16}
        result = manager_cuda.bulk_get_embeddings(seqs, methods=["mean", "max", "last_token"])

        for method in ["mean", "max", "last_token"]:
            vec = result["s1"][method]
            assert len(vec) >= 64
            assert not all(v == 0.0 for v in vec), f"{method}: all zeros"
            print(f"  {method}: dim={len(vec)}, sum={sum(v for v in vec[:10]):.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
