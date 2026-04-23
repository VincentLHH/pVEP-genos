def test_embedding_cache_reuse():
    from models.embedding_manager import EmbeddingManager

    manager = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path="/home/share/huadjyin/home/liuhaohan/genos/Genos-1.2B",
        device="cuda",
        batch_size=2
    )

    seqs1 = {
        "a": "ACGTACGTACGT",
        "b": "TTTTTTTTTTTT"
    }

    print("\n--- First run (should compute) ---")
    result1 = manager.get_embeddings(seqs1, methods=["mean"])

    cache_size_after_first = len(manager.cache)

    seqs2 = {
        "c": "ACGTACGTACGT",  # same as a
        "d": "TTTTTTTTTTTT"   # same as b
    }

    print("\n--- Second run (should hit cache) ---")
    result2 = manager.get_embeddings(seqs2, methods=["mean"])

    cache_size_after_second = len(manager.cache)

    # 🔥 cache 不应该增长
    assert cache_size_after_first == cache_size_after_second

    # 🔥 embedding 一致
    assert result1["a"]["mean"] == result2["c"]["mean"]
    assert result1["b"]["mean"] == result2["d"]["mean"]

    print("✅ Cache reuse test passed")
