def test_embedding_manager_basic():
    from models.embedding_manager import EmbeddingManager

    manager = EmbeddingManager(
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
