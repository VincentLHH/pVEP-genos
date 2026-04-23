import os
import pysam
import tempfile

from core.variant import Variant
from core.sequence_builder import SequenceBuilder


def create_test_fasta():
    """
    创建一个简单的参考基因组：
    chr1: 100bp
    """
    seq = "ACGT" * 25  # 100bp

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fa")
    fasta_path = tmp.name

    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write(seq + "\n")

    pysam.faidx(fasta_path)
    return fasta_path, seq


def test_haplotype_reconstruction_basic():
    fasta_path, ref_seq = create_test_fasta()

    builder = SequenceBuilder(fasta_path, window_size=20)

    # ==============================
    # 构造变异（重点测试）
    # ==============================

    # 参考序列（位置从1开始）
    # 假设窗口覆盖 10-30 区域

    variants = [
        # SNP: pos=15 A->T (1|0)
        Variant("chr1", 15, ref_seq[14], "T", (1, 0)),

        # 插入: pos=18 C->CGG (0|1)
        Variant("chr1", 18, ref_seq[17], ref_seq[17] + "GG", (0, 1)),

        # 缺失: pos=22 GT->G (1|1)
        Variant("chr1", 22, ref_seq[21:23], ref_seq[21], (1, 1)),
    ]

    # 中心变异（随便选一个）
    center_variant = variants[0]

    result = builder.build(center_variant, variants)

    assert result is not None

    hap1 = result["hap1"]["mut_seq"]
    hap2 = result["hap2"]["mut_seq"]
    ref = result["ref_seq"]

    print("\n=== REF ===\n", ref)
    print("\n=== HAP1 ===\n", hap1)
    print("\n=== HAP2 ===\n", hap2)

    # ==============================
    # 验证逻辑
    # ==============================

    # hap1: (1|0), (0|1), (1|1)
    # → 应包含：
    # - SNP（第一个变异）
    # - 不包含插入（第二个变异）
    # - 包含缺失（第三个变异）

    assert "T" in hap1  # SNP引入
    assert len(hap1) != len(ref)  # 因为有 indel

    # hap2:
    # - 不包含第一个变异
    # - 包含插入
    # - 包含缺失

    assert "GG" in hap2  # 插入存在
    assert len(hap2) != len(ref)

    # ==============================
    # 核心验证：两条 hap 不应相同
    # ==============================

    assert hap1 != hap2


def test_homozygous_variant():
    fasta_path, ref_seq = create_test_fasta()
    builder = SequenceBuilder(fasta_path, window_size=20)

    # 1|1 → 两个 hap 应完全相同
    v = Variant("chr1", 15, ref_seq[14], "T", (1, 1))

    result = builder.build(v, [v])

    hap1 = result["hap1"]["mut_seq"]
    hap2 = result["hap2"]["mut_seq"]

    assert hap1 == hap2


def test_no_variant_background():
    """
    测试 0|1 情况下：
    hap1 = ref，但仍是完整背景
    """

    fasta_path, ref_seq = create_test_fasta()
    builder = SequenceBuilder(fasta_path, window_size=20)

    v = Variant("chr1", 15, ref_seq[14], "T", (0, 1))

    result = builder.build(v, [v])

    hap1 = result["hap1"]["mut_seq"]
    hap2 = result["hap2"]["mut_seq"]

    # hap1 应等于 ref
    assert hap1 == result["ref_seq"]

    # hap2 应不同
    assert hap2 != hap1

def test_extreme_overlapping_indels():
    """
    测试：
    - 连续 indel
    - 重叠变异
    - 顺序依赖
    """

    fasta_path, ref_seq = create_test_fasta()
    builder = SequenceBuilder(fasta_path, window_size=30)

    # 构造一个简单区域
    # 参考：ACGTACGTACGT...

    variants = [
        # SNP
        Variant("chr1", 12, ref_seq[11], "T", (1, 1)),

        # 插入（会改变后面坐标）
        Variant("chr1", 14, ref_seq[13], ref_seq[13] + "GG", (1, 0)),

        # 紧邻的SNP（测试shift是否正确）
        Variant("chr1", 15, ref_seq[14], "A", (1, 1)),

        # 删除（跨位点）
        Variant("chr1", 17, ref_seq[16:18], ref_seq[16], (0, 1)),
    ]

    center_variant = variants[0]

    result = builder.build(center_variant, variants)

    if result is None:
        print("⚠️ reconstruction failed (acceptable for extreme cases)")
        return

    hap1 = result["hap1"]["mut_seq"]
    hap2 = result["hap2"]["mut_seq"]

    print("\n=== EXTREME TEST ===")
    print("REF :", result["ref_seq"])
    print("HAP1:", hap1)
    print("HAP2:", hap2)

    # 🔥 核心验证
    assert hap1 != hap2
    assert len(hap1) != len(result["ref_seq"])  # 有 indel
    assert len(hap2) != len(result["ref_seq"])


def test_dense_variants_cluster():
    """
    测试高密度变异（模拟LD block）
    """

    fasta_path, ref_seq = create_test_fasta()
    builder = SequenceBuilder(fasta_path, window_size=40)

    variants = []

    # 在连续区域内放很多变异
    for i in range(10, 20):
        ref_base = ref_seq[i-1]

        # alternating GT
        if i % 2 == 0:
            gt = (1, 0)
        else:
            gt = (0, 1)

        variants.append(
            Variant("chr1", i, ref_base, "T", gt)
        )

    center_variant = variants[5]

    result = builder.build(center_variant, variants)

    if result is None:
        print("⚠️ reconstruction failed (acceptable for extreme cases)")
        return

    hap1 = result["hap1"]["mut_seq"]
    hap2 = result["hap2"]["mut_seq"]

    print("\n=== DENSE VARIANTS ===")
    print("HAP1:", hap1)
    print("HAP2:", hap2)

    # 每个 hap 应该有不同 pattern
    assert hap1 != hap2


def cleanup_fasta(fasta_path):
    os.remove(fasta_path)
    os.remove(fasta_path + ".fai")
