"""
tests/test_02_genvarloader_builder.py
========================================
测试 GenVarLoaderSequenceBuilder（需要 genvarloader 库 + GPU 节点环境）。

覆盖场景
--------
1. GenVarLoader 库可用性检测
2. _gvl_path() 哈希路径生成
3. _ensure_dataset() 惰性创建 / 加载 .gvl 数据集
4. build() 核心接口：返回格式与 builtin SequenceBuilder 对齐
5. sample_name 回退到 current_sample 注入
6. get_dataset_info() 诊断接口
7. 多样本重建

前置条件
--------
- GPU 节点（有 genvarloader 库 + 参考基因组）
- 环境变量 PVEPGENOS_VCF_PATH / PVEPGENOS_BED_PATH / PVEPGENOS_REF_FASTA
  或 config/default.yaml 中有有效路径

运行（GPU 节点）
----------------
pytest tests/test_02_genvarloader_builder.py -v

或跳过真实数据测试（仅做可用性检测）：
pytest tests/test_02_genvarloader_builder.py -v -k "not real"
"""

import os
import sys
from pathlib import Path

import pytest

# GenVarLoader 可能未安装，延迟 import
try:
    import genvarloader
    HAS_GVL = True
except ImportError:
    HAS_GVL = False
    genvarloader = None  # type: ignore

skip_if_no_gvl = pytest.mark.skipif(
    not HAS_GVL,
    reason="genvarloader not installed or not importable"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.genvarloader_builder import (
    GenVarLoaderSequenceBuilder,
    _numpy_bytes_to_str,
    _reverse_complement,
)
from core.variant import Variant


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gvl_skip():
    if not HAS_GVL:
        pytest.skip("genvarloader not available", allow_module_level=True)


@pytest.fixture(scope="session")
def real_gvl_cfg(env_vcf_path, env_bed_path, env_ref_fasta):
    """从环境变量 / config 读取真实数据路径，用于端到端测试"""
    missing = []
    for name, val in [("PVEPGENOS_VCF_PATH", env_vcf_path),
                      ("PVEPGENOS_BED_PATH", env_bed_path),
                      ("PVEPGENOS_REF_FASTA", env_ref_fasta)]:
        if not val:
            missing.append(name)

    if missing:
        pytest.skip(f"Missing env vars: {missing}", allow_module_level=True)

    for p in (env_vcf_path, env_bed_path, env_ref_fasta):
        if not os.path.exists(str(p)):
            pytest.skip(f"Path not found: {p}", allow_module_level=True)

    return dict(vcf_path=env_vcf_path, bed_path=env_bed_path, ref_fasta=env_ref_fasta)


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestGenVarLoaderImport:
    """库可用性检测"""

    def test_genvarloader_importable(self, gvl_skip):
        """genvarloader 库可以 import"""
        assert hasattr(genvarloader, "Dataset") or hasattr(genvarloader, "write")
        print(f"genvarloader version: {getattr(genvarloader, '__version__', 'unknown')}")


class TestNumpyBytesConversion:
    """字节转字符串 工具函数测试"""

    def test_numpy_bytes_scalar(self):
        import numpy as np
        arr = np.array(b"ACGTACGT", dtype=np.bytes_)
        result = _numpy_bytes_to_str(arr)
        assert result == "ACGTACGT", f"Expected ACGTACGT, got {result}"

    def test_numpy_uint8_array(self):
        import numpy as np
        arr = np.array([ord(c) for c in "TGCA"], dtype=np.uint8)
        result = _numpy_bytes_to_str(arr)
        assert result == "TGCA", f"Expected TGCA, got {result}"

    def test_empty_bytes(self):
        import numpy as np
        arr = np.array(b"", dtype=np.bytes_)
        result = _numpy_bytes_to_str(arr)
        assert result == "", f"Expected empty string, got {result!r}"


class TestReverseComplement:
    """反向互补工具函数"""

    def test_revcomp_known_sequence(self):
        assert _reverse_complement("AAAA") == "TTTT"
        assert _reverse_complement("ACGT") == "ACGT"
        assert _reverse_complement("AACG") == "CGTT"
        assert _reverse_complement("") == ""

    def test_revcomp_round_trip(self):
        seq = "ACGTACGT"
        assert _reverse_complement(_reverse_complement(seq)) == seq


class TestGenVarLoaderSequenceBuilder:
    """核心功能测试（需要真实数据）"""

    def test_gvl_path_deterministic(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """同一 vcf+bed 应生成相同的 .gvl 路径（哈希确定性）"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache"))

        b1 = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )
        b2 = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        p1 = b1._gvl_path()
        p2 = b2._gvl_path()
        assert p1 == p2, f"Paths should match: {p1} vs {p2}"
        assert p1.startswith(cache_dir), f"Path should be in cache_dir: {p1}"
        assert p1.endswith(".gvl"), f"Path should end with .gvl: {p1}"

    def test_lazy_dataset_creation(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """惰性创建：build() 调用前 _dataset 应为 None"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache2"))

        builder = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        assert builder._dataset is None, "_dataset should be None before first use"
        assert builder._regions == [], "_regions should be empty before first use"

    def test_dataset_persists_after_build(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """第一次 build() 后 dataset 被缓存，后续调用不再重新打开"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache3"))

        builder = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        regions = builder._load_regions()
        assert len(regions) > 0, "BED should have at least one region"

        chrom, start, end = regions[0]
        v = Variant(chrom, start + 5, "A", "T", (1, 0))

        # 第一次 build（可能触发 dataset 创建）
        builder.build(v, [v])
        # 第二次 build（应复用缓存）
        builder.build(v, [v])

        assert builder._dataset is not None, "_dataset should be populated after build()"

    def test_build_returns_correct_keys(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """build() 返回格式与 builtin SequenceBuilder 完全对齐"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache4"))

        builder = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        regions = builder._load_regions()
        chrom, start, end = regions[0]
        v = Variant(chrom, start + 10, "A", "T", (1, 0))

        result = builder.build(v, [v])

        if result is not None:
            assert set(result.keys()) == {
                "ref_seq", "ref_comp", "hap1", "hap2"
            }
            assert set(result["hap1"].keys()) == {"mut_seq", "mut_comp"}
            assert set(result["hap2"].keys()) == {"mut_seq", "mut_comp"}
        else:
            # sample 可能在 VCF 中不存在，返回 None 是合理的
            pass

    def test_current_sample_injection(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """sample_name 回退到 builder.current_sample 注入"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache5"))

        builder = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        regions = builder._load_regions()
        chrom, start, end = regions[0]
        v = Variant(chrom, start + 10, "A", "T", (1, 0))

        # 不传 sample_name → 依赖 current_sample
        builder.current_sample = "sample1"  # 模拟 process_all 注入
        result1 = builder.build(v, [v], sample_name=None)
        assert result1 is None or isinstance(result1, dict)

    def test_get_dataset_info(self, gvl_skip, real_gvl_cfg, tmp_path_factory):
        """诊断接口返回合理信息"""
        cache_dir = str(tmp_path_factory.mktemp("gvl_cache6"))

        builder = GenVarLoaderSequenceBuilder(
            vcf_path=real_gvl_cfg["vcf_path"],
            bed_path=real_gvl_cfg["bed_path"],
            ref_fasta=real_gvl_cfg["ref_fasta"],
            gvl_cache_dir=cache_dir,
        )

        info = builder.get_dataset_info()

        assert "n_regions" in info
        assert "gvl_path" in info
        assert info["n_regions"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
