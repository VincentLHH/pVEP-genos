def test_sample_pipeline():
    from core.sample import Sample
    from core.variant import Variant
    from core.sequence_builder import SequenceBuilder
    from models.embedding_manager import EmbeddingManager

    # =========================
    # 1️⃣ 构造一个小 reference fasta
    # =========================
    import tempfile

    fasta_content = ">chr1\nACGTACGTACGTACGTACGTACGTACGTACGT"
    tmp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fa")

    with open(tmp_fasta.name, "w") as f:
        f.write(fasta_content)

    # =========================
    # 2️⃣ 构造 variants（简单 case）
    # =========================
    variants = [
        Variant("chr1", 10, "C", "T", (1, 0)),
        Variant("chr1", 15, "A", "G", (0, 1)),
    ]

    # =========================
    # 3️⃣ 初始化模块
    # =========================
    builder = SequenceBuilder(tmp_fasta.name, window_size=16)

    manager = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path="/home/share/huadjyin/home/liuhaohan/genos/Genos-1.2B",
        device="cuda"
    )

    sample = Sample("test_sample", "test_outputs")

    # =========================
    # 4️⃣ 跑 pipeline
    # =========================
    sample.process_all(
        variants,
        builder,
        manager,
        methods=["mean"]
    )

    # =========================
    # 5️⃣ 验证结果
    # =========================
    assert len(sample.embeddings) > 0

    for var_id, models in sample.embeddings.items():
        assert "Genos-1.2B" in models

    print("✅ Sample pipeline test passed")
