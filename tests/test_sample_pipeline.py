"""
tests/test_sample_pipeline.py (旧版)
========================================
Sample pipeline 端到端测试。

需要 GPU + 模型权重，CPU 环境自动 skip。

运行
----
pytest tests/test_sample_pipeline.py -v
"""

import pytest
from tests.conftest import skip_if_no_cuda_or_model


@skip_if_no_cuda_or_model
def test_sample_pipeline():
    from core.sample import Sample
    from core.variant import Variant
    from core.sequence_builder import SequenceBuilder
    from models.embedding_manager import EmbeddingManager

    # =========================
    # 构造一个小 reference fasta
    # =========================
    import tempfile
    import os
    import pysam

    fasta_content = ">chr1\nACGTACGTACGTACGTACGTACGTACGTACGT"
    tmp_fasta = tempfile.NamedTemporaryFile(delete=False, suffix=".fa")

    with open(tmp_fasta.name, "w") as f:
        f.write(fasta_content)

    try:
        pysam.faidx(tmp_fasta.name)
    except Exception:
        pass

    # =========================
    # 构造 variants（简单 case）
    # =========================
    variants = [
        Variant("chr1", 10, "C", "T", (1, 0)),
        Variant("chr1", 15, "A", "G", (0, 1)),
    ]

    # =========================
    # 初始化模块
    # =========================
    builder = SequenceBuilder(tmp_fasta.name, window_size=16)

    manager = EmbeddingManager(
        model_name="Genos-1.2B",
        model_path="/home/share/huadjyin/home/liuhaohan/genos/Genos-1.2B",
        device="cuda"
    )

    sample = Sample("test_sample", "test_outputs")

    # =========================
    # 跑 pipeline
    # =========================
    sample.process_all(
        variants,
        builder,
        manager,
        methods=["mean"]
    )

    # =========================
    # 验证结果
    # =========================
    assert len(sample.embeddings) > 0

    for var_id, models in sample.embeddings.items():
        assert "Genos-1.2B" in models

    # Cleanup
    try:
        os.remove(tmp_fasta.name)
        for ext in (".fai",):
            try:
                os.remove(tmp_fasta.name + ext)
            except FileNotFoundError:
                pass
    except Exception:
        pass

    print("✅ Sample pipeline test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
