"""
tests/test_embedding_manager.py (旧版)
========================================
EmbeddingManager 基础功能测试。

需要 GPU + 模型权重，CPU 环境自动 skip。

运行
----
pytest tests/test_embedding_manager.py -v
"""

import pytest
from tests.conftest import skip_if_no_cuda_or_model


@skip_if_no_cuda_or_model
def test_embedding_manager_basic():
    from models.embedding_manager import EmbeddingManager

    manager = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path="/home/share/huadjyin/home/liuhaohan/genos/Genos-1.2B",
        device="cuda"
    )

    seqs = {
        "a": "ACGTACGTACGT",
        "b": "ACGTACGTACGT",  # duplicate
        "c": "TTTTTTTTTTTT"
    }

    result = manager.get_embeddings(seqs, methods=["mean"])

    assert "a" in result
    assert "b" in result

    # 去重验证
    assert result["a"]["mean"] == result["b"]["mean"]

    print("✅ EmbeddingManager test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
