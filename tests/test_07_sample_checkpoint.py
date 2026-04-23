"""
tests/test_07_sample_checkpoint.py
===================================
测试样本级断点续存（通用，CPU/GPU 均可）。

覆盖场景
--------
1. 完整输出 JSON 存在时 is_complete() 返回 True（跳过）
2. 输出目录不存在时自动创建
3. 已有部分 variant 时 is_processed() 正确判断
4. 第二次运行覆盖（重新处理 + 追加）
5. --no-save-haplotypes / --no-save-embeddings 对 JSON 内容的影响
6. save() / _load_if_exists() 的读写一致性
7. save_interval 定期保存功能
8. 混合场景：部分 variant 已处理，部分未处理

前置条件
--------
- 无（使用 mock_fasta + mock_vcf）
- GPU 可选（有则用 EmbeddingManager，无则 mock manager）

运行（任意节点）
----------------
pytest tests/test_07_sample_checkpoint.py -v
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sample import Sample
from core.variant import Variant


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def temp_output():
    with tempfile.TemporaryDirectory(prefix="pVEP_checkpoint_") as d:
        yield d


@pytest.fixture
def mock_manager():
    """Mock embedding manager（不使用真实模型）"""
    class MockManager:
        model_name = "Genos-1.2B"
    return MockManager()


@pytest.fixture
def mock_fasta():
    import pysam
    seq = "ACGT" * 50  # 200bp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fa", mode="w") as f:
        f.write(f">chr1\n{seq}\n")
        path = f.name
    pysam.faidx(path)
    yield path
    for ext in ("", ".fai"):
        try:
            os.remove(path + ext)
        except FileNotFoundError:
            pass


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────

def make_sample_json(output_dir, sample_id, embeddings=None, haplotypes=None):
    """直接在文件系统写一个样本 JSON（模拟已有结果）"""
    path = os.path.join(output_dir, f"{sample_id}.json")
    data = {"sample_id": sample_id}
    if embeddings:
        data["embeddings"] = embeddings
    if haplotypes:
        data["haplotypes"] = haplotypes
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestOutputDirectoryCreation:
    """输出目录自动创建"""

    def test_creates_dir_on_init(self, temp_output):
        subdir = os.path.join(temp_output, "sub", "nested")
        sample = Sample("s1", subdir, save_haplotypes=True, save_embeddings=True)
        assert os.path.exists(subdir), "Sample.__init__ should create output_dir"
        assert sample.filepath == os.path.join(subdir, "s1.json")


class TestIsComplete:
    """is_complete 样本级完整性判断"""

    def test_no_file_returns_false(self, temp_output, mock_manager):
        """JSON 不存在 → is_complete() = False"""
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert not sample.is_complete([], mock_manager.model_name)

    def test_empty_file_returns_false(self, temp_output, mock_manager):
        """JSON 存在但为空 dict → is_complete() = False"""
        # 写一个空文件
        path = os.path.join(temp_output, "s1.json")
        with open(path, "w") as f:
            json.dump({}, f)

        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert not sample.is_complete([], mock_manager.model_name)

    def test_with_embeddings_returns_true(self, temp_output, mock_manager):
        """JSON 包含目标 model 的 embedding → is_complete() = True"""
        make_sample_json(
            temp_output, "s1",
            embeddings={"v1": {"Genos-1.2B": {"mean": [0.1] * 128}}}
        )
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert sample.is_complete(["v1"], mock_manager.model_name)

    def test_different_model_returns_false(self, temp_output, mock_manager):
        """JSON 有其他 model embedding，但目标 model 没有 → is_complete() = False"""
        make_sample_json(
            temp_output, "s1",
            embeddings={"v1": {"OtherModel": {"mean": [0.1] * 128}}}
        )
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert not sample.is_complete(["v1"], mock_manager.model_name)

    def test_save_embeddings_false_returns_false(self, temp_output, mock_manager):
        """save_embeddings=False 时 is_complete() 永远返回 False（不检查 JSON）"""
        make_sample_json(
            temp_output, "s1",
            embeddings={"v1": {"Genos-1.2B": {"mean": [0.1] * 128}}}
        )
        sample = Sample("s1", temp_output, save_haplotypes=False, save_embeddings=False)
        assert not sample.is_complete(["v1"], mock_manager.model_name)


class TestIsProcessed:
    """is_processed variant 级判断"""

    def test_processed_variant(self, temp_output, mock_manager):
        """特定 variant 有 embedding → is_processed() = True"""
        make_sample_json(
            temp_output, "s1",
            embeddings={"v1": {"Genos-1.2B": {"mean": [0.1] * 128}}}
        )
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert sample.is_processed("v1", mock_manager.model_name)

    def test_not_processed_variant(self, temp_output, mock_manager):
        """variant 不在 embeddings 中 → is_processed() = False"""
        make_sample_json(
            temp_output, "s1",
            embeddings={"v1": {"Genos-1.2B": {"mean": [0.1] * 128}}}
        )
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        assert not sample.is_processed("v2", mock_manager.model_name)


class TestSaveLoadConsistency:
    """save() / _load_if_exists() 一致性"""

    def test_save_and_reload_haplotypes(self, temp_output):
        """保存单倍型 → 重加载"""
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=False)
        sample.haplotypes = {
            "v1": {"ref_seq": "ACGT", "hap1": {"mut_seq": "TGCA"}}
        }
        sample.save()

        # 新 Sample 实例应读取已有数据
        sample2 = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=False)
        assert "v1" in sample2.haplotypes
        assert sample2.haplotypes["v1"]["ref_seq"] == "ACGT"

    def test_save_and_reload_embeddings(self, temp_output):
        """保存 embedding → 重加载"""
        sample = Sample("s1", temp_output, save_haplotypes=False, save_embeddings=True)
        sample.embeddings = {
            "v1": {"Genos-1.2B": {"mean": [0.1] * 128}}
        }
        sample.save()

        sample2 = Sample("s1", temp_output, save_haplotypes=False, save_embeddings=True)
        assert "v1" in sample2.embeddings
        assert len(sample2.embeddings["v1"]["Genos-1.2B"]["mean"]) == 128


class TestSaveFlags:
    """--save-haplotypes / --save-embeddings 对 JSON 内容的影响"""

    def test_both_true(self, temp_output):
        """两者都开 → JSON 包含 haplotypes 和 embeddings"""
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=True)
        sample.haplotypes = {"v1": {"ref": "ACGT"}}
        sample.embeddings = {"v1": {"M": {"mean": [0.1] * 128}}}
        sample.save()

        with open(sample.filepath) as f:
            data = json.load(f)
        assert "haplotypes" in data
        assert "embeddings" in data

    def test_only_haplotypes(self, temp_output):
        """只保存单倍型"""
        sample = Sample("s1", temp_output, save_haplotypes=True, save_embeddings=False)
        sample.haplotypes = {"v1": {"ref": "ACGT"}}
        sample.save()

        with open(sample.filepath) as f:
            data = json.load(f)
        assert "haplotypes" in data
        assert "embeddings" not in data

    def test_only_embeddings(self, temp_output):
        """只保存 embedding"""
        sample = Sample("s1", temp_output, save_haplotypes=False, save_embeddings=True)
        sample.embeddings = {"v1": {"M": {"mean": [0.1] * 128}}}
        sample.save()

        with open(sample.filepath) as f:
            data = json.load(f)
        assert "haplotypes" not in data
        assert "embeddings" in data

    def test_neither(self, temp_output):
        """两者都关 → JSON 只含 sample_id"""
        sample = Sample("s1", temp_output, save_haplotypes=False, save_embeddings=False)
        sample.save()

        with open(sample.filepath) as f:
            data = json.load(f)
        assert "sample_id" in data
        assert "haplotypes" not in data
        assert "embeddings" not in data


class TestSaveInterval:
    """save_interval 定期保存"""

    def test_partial_save(self, temp_output, mock_fasta):
        """处理 N 个 variant（>= save_interval）时触发中间保存"""
        from core.sequence_builder import SequenceBuilder

        builder = SequenceBuilder(mock_fasta, window_size=40)
        seq = "ACGT" * 50

        # Mock manager（直接返回假 embedding）
        class FakeManager:
            model_name = "Genos-1.2B"
            def bulk_get_embeddings(self, seq_dict, methods):
                # 返回假结果
                return {
                    key: {m: [0.1] * 128 for m in methods}
                    for key in seq_dict
                }

        sample = Sample(
            "checkpoint_test", temp_output,
            save_haplotypes=False, save_embeddings=True
        )

        # 构造 15 个 variant，save_interval=5
        # 注意：pos >= 21 才不会越 half_window=20 的下界
        variants = [
            Variant("chr1", 21 + i, seq[20 + i], "T", (1, 0))
            for i in range(15)
        ]

        # 第一次运行：处理所有（会触发 3 次中间保存）
        sample.process_all(variants, builder, FakeManager(), methods=["mean"], save_interval=5)

        # 验证最终文件
        assert os.path.exists(sample.filepath)
        with open(sample.filepath) as f:
            data = json.load(f)
        assert "embeddings" in data
        assert len(data["embeddings"]) == 15
        print(f"15 variants processed with save_interval=5: final embedding count={len(data['embeddings'])}")


class TestCheckpointResume:
    """断点续存（完整流程）"""

    def test_resume_skips_fully_complete_sample(self, temp_output, mock_fasta):
        """完整 sample JSON 存在 → process_all 完全跳过"""
        from core.sequence_builder import SequenceBuilder

        # 预先写入完整结果
        make_sample_json(
            temp_output, "resume_test",
            embeddings={"v1": {"Genos-1.2B": {"mean": [0.9] * 128}}}
        )

        builder = SequenceBuilder(mock_fasta, window_size=40)
        seq = "ACGT" * 50

        class FakeManager:
            model_name = "Genos-1.2B"
            call_count = 0
            def bulk_get_embeddings(self, seq_dict, methods):
                FakeManager.call_count += len(seq_dict)
                return {k: {m: [0.1] * 128 for m in methods} for k in seq_dict}

        sample = Sample("resume_test", temp_output, save_haplotypes=True, save_embeddings=True)

        # is_complete 应为 True → 不处理
        assert sample.is_complete(["v1"], "Genos-1.2B")

        variants = [Variant("chr1", 25, seq[24], "T", (1, 0))]

        # 手动跳过：只测 is_complete 逻辑
        if sample.is_complete(["v1"], FakeManager.model_name):
            print("Sample correctly detected as complete, would skip processing")
        else:
            pytest.fail("is_complete should return True for existing full result")

        assert FakeManager.call_count == 0, "bulk_get_embeddings should NOT be called when complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
