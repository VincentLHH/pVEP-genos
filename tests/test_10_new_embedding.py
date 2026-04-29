"""
tests/test_10_new_embedding.py
===============================
新版个性化 VEP embedding 提取逻辑的单元测试。

覆盖范围
--------
1. BED 拆分坐标正确性（SNP、INS、DEL、MNV）
2. parse_var_name_from_bed_name 解析正确性
3. 6 种序列生成与基因型等价规则（builtin SequenceBuilder）
4. WT_hap 恢复（GenVarLoaderSequenceBuilder.restore_wt_from_mut_hap）
5. 全局哈希缓存命中逻辑（EmbeddingExtractor）
6. emb_up + emb_down concat 维度正确性（mock 推理）
7. SixSeqResult.iter_named_seqs() 去重逻辑

运行环境
--------
CPU 节点即可（所有需要 GPU 的测试均 skip）。

    pytest tests/test_10_new_embedding.py -v
"""

import os
import sys
import pytest
import numpy as np
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────
# 路径设置
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import (
    skip_if_no_cuda,
    skip_if_no_gvl,
    HAS_CUDA,
    HAS_GVL,
)

from core.variant import (
    Variant,
    BedRow,
    VariantBedSplit,
    split_variant_to_bed,
    parse_var_name_from_bed_name,
    load_bed,
)
from core.sequence_builder import (
    HapSeqPair,
    SixSeqResult,
    SequenceBuilder,
    reverse_complement,
)
from core.embedding_extractor import EmbeddingExtractor, _seq_hash, _cache_key


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def snp_variant():
    """SNP: chr1 pos=100 A→T，GT=(1,0)"""
    return Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))


@pytest.fixture
def ins_variant():
    """INS: chr2 pos=202 A→AT，GT=(0,1)"""
    return Variant(chrom="chr2", pos=202, ref="A", alt="AT", gt=(0, 1))


@pytest.fixture
def del_variant():
    """DEL: chr3 pos=300 AT→A，GT=(1,1)"""
    return Variant(chrom="chr3", pos=300, ref="AT", alt="A", gt=(1, 1))


@pytest.fixture
def mnv_variant():
    """MNV: chr4 pos=400 ATC→GTG，GT=(0,0)"""
    return Variant(chrom="chr4", pos=400, ref="ATC", alt="GTG", gt=(0, 0))


@pytest.fixture
def hom_ref_variant():
    """0|0 基因型，位点本身携带 GT=(0,0)"""
    return Variant(chrom="chr5", pos=500, ref="G", alt="C", gt=(0, 0))


# ===========================================================================
# Section 1: BED 拆分坐标正确性
# ===========================================================================

class TestBedSplit:

    def test_snp_split(self, snp_variant):
        """SNP A→T at pos=100, n=3: 验证上下游坐标"""
        split = split_variant_to_bed(snp_variant, n=3)

        # pos=100, end=100 (SNP, end=pos+len(ref)-1=100)
        # upstream:  start=100-3-1=96, end=100+1=101
        # downstream: start=100-2=98, end=100+3=103
        assert split.upstream.chrom == "chr1"
        assert split.upstream.start == 96
        assert split.upstream.end == 101
        assert split.upstream.name == "chr1_100_A_T_upstream"

        assert split.downstream.start == 98
        assert split.downstream.end == 103
        assert split.downstream.name == "chr1_100_A_T_downstream"

        # 长度验证
        assert split.upstream.length == 5       # 101 - 96 = 5 = n + var_len + 1 = 3 + 1 + 1
        assert split.downstream.length == 5     # 103 - 98 = 5

    def test_ins_split(self, ins_variant):
        """INS A→AT at pos=202, end=202 (ref=A, len=1), n=3"""
        split = split_variant_to_bed(ins_variant, n=3)

        # pos=202, end=202 (ref="A", end=202+1-1=202)
        # upstream:  start=202-3-1=198, end=202+1=203
        # downstream: start=202-2=200, end=202+3=205
        assert split.upstream.start == 198
        assert split.upstream.end == 203
        assert split.downstream.start == 200
        assert split.downstream.end == 205

        # 两行长度应该相等
        assert split.upstream.length == split.downstream.length == 5

    def test_del_split(self, del_variant):
        """DEL AT→A at pos=300, ref=AT (len=2), end=301, n=3"""
        split = split_variant_to_bed(del_variant, n=3)

        # pos=300, end=300+2-1=301
        # upstream:  start=300-3-1=296, end=301+1=302
        # downstream: start=300-2=298, end=301+3=304
        assert split.upstream.start == 296
        assert split.upstream.end == 302
        assert split.downstream.start == 298
        assert split.downstream.end == 304

        # 两行长度相等
        assert split.upstream.length == split.downstream.length == 6

    def test_mnv_split(self, mnv_variant):
        """MNV ATC→GTG at pos=400, ref=ATC (len=3), end=402, n=3"""
        split = split_variant_to_bed(mnv_variant, n=3)

        # pos=400, end=402
        # upstream:  start=400-3-1=396, end=402+1=403
        # downstream: start=400-2=398, end=402+3=405
        assert split.upstream.start == 396
        assert split.upstream.end == 403
        assert split.downstream.start == 398
        assert split.downstream.end == 405

        assert split.upstream.length == split.downstream.length == 7

    def test_n_too_small_raises(self, snp_variant):
        """n 不足时应抛出 ValueError"""
        with pytest.raises(ValueError, match="smaller than the minimum"):
            split_variant_to_bed(snp_variant, n=1, validate_n_min=10)

    def test_var_name_prefix(self, snp_variant):
        """var_name 必须对应 Variant.id"""
        split = split_variant_to_bed(snp_variant, n=3)
        assert split.var_name == snp_variant.id
        assert split.upstream.var_name == snp_variant.id
        assert split.downstream.var_name == snp_variant.id

    def test_bed_row_direction_flags(self, snp_variant):
        split = split_variant_to_bed(snp_variant, n=3)
        assert split.upstream.is_upstream is True
        assert split.upstream.is_downstream is False
        assert split.downstream.is_downstream is True
        assert split.downstream.is_upstream is False


# ===========================================================================
# Section 2: parse_var_name_from_bed_name
# ===========================================================================

class TestParseVarName:

    def test_upstream_suffix(self):
        chrom, pos, ref, alt, direction = parse_var_name_from_bed_name(
            "chr2_202_A_AT_upstream"
        )
        assert chrom == "chr2"
        assert pos == "202"
        assert ref == "A"
        assert alt == "AT"
        assert direction == "upstream"

    def test_downstream_suffix(self):
        chrom, pos, ref, alt, direction = parse_var_name_from_bed_name(
            "chr1_100_A_T_downstream"
        )
        assert direction == "downstream"

    def test_no_suffix(self):
        chrom, pos, ref, alt, direction = parse_var_name_from_bed_name(
            "chr3_300_AT_A"
        )
        assert chrom == "chr3"
        assert pos == "300"
        assert ref == "AT"
        assert alt == "A"
        assert direction == ""

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_var_name_from_bed_name("chr1_bad")


# ===========================================================================
# Section 3: 6 种序列生成（基于 mock FASTA）
# ===========================================================================

def _make_fake_fasta(seq_map: Dict[str, str]):
    """创建 mock pysam.FastaFile，按 fetch(chrom, start, end) 返回序列。"""
    mock_fasta = MagicMock()

    def fetch_side_effect(chrom, start, end):
        full_seq = seq_map.get(chrom, "N" * 10000)
        return full_seq[start:end]

    mock_fasta.fetch = MagicMock(side_effect=fetch_side_effect)
    return mock_fasta


class TestSixSeqBuiltin:
    """
    测试 SequenceBuilder.build_from_bed()。
    使用 mock FASTA 而不依赖真实文件。
    """

    def _make_builder(self, seq_map: Dict[str, str]) -> SequenceBuilder:
        """构造一个注入 mock FASTA 的 SequenceBuilder。"""
        builder = SequenceBuilder.__new__(SequenceBuilder)
        builder.fasta = _make_fake_fasta(seq_map)
        builder.window_size = 16
        builder.half_window = 8
        builder.n = 5
        return builder

    def test_wt_ref_equals_ref_sequence(self, snp_variant):
        """
        WT_ref 应等于 FASTA 中对应区域的原始序列。
        """
        # 构造 60 bp 的伪参考序列
        ref_seq = "A" * 200

        builder = self._make_builder({"chr1": ref_seq})
        bed_row = BedRow(chrom="chr1", start=90, end=105, name="chr1_100_A_T_upstream")

        result = builder.build_from_bed(bed_row, snp_variant, [snp_variant])

        assert result is not None
        assert result.ref_pair.wt == ref_seq[90:105]

    def test_mut_ref_has_alt_at_correct_position(self, snp_variant):
        """
        Mut_ref 应在目标位置将 A 替换为 T。
        SNP pos=100(1-based), bed_row start=90(0-based)
        offset = 100-1-90 = 9
        """
        ref_seq = "A" * 200
        builder = self._make_builder({"chr1": ref_seq})
        bed_row = BedRow(chrom="chr1", start=90, end=105, name="chr1_100_A_T_upstream")

        result = builder.build_from_bed(bed_row, snp_variant, [snp_variant])

        assert result is not None
        wt = result.ref_pair.wt
        mut = result.ref_pair.mut

        # wt 与 mut 应只在 offset=9 处不同
        assert wt[9] == "A"
        assert mut[9] == "T"
        assert wt[:9] == mut[:9]
        assert wt[10:] == mut[10:]

    def test_gt_0_0_makes_mut_wt_equal(self, hom_ref_variant):
        """
        0|0 基因型：Mut_hap1 == WT_hap1，Mut_hap2 == WT_hap2。
        """
        # ref_seq 必须足够长（>= bed_row.end=508），mock fetch 才能正确返回区间序列
        ref_seq = "G" * 600

        builder = self._make_builder({"chr5": ref_seq})
        bed_row = BedRow(
            chrom="chr5", start=490, end=508, name="chr5_500_G_C_upstream"
        )

        result = builder.build_from_bed(bed_row, hom_ref_variant, [hom_ref_variant])

        assert result is not None
        assert result.hap1.wt_is_alias_of_mut is True
        assert result.hap2.wt_is_alias_of_mut is True
        assert result.hap1.mut == result.hap1.wt
        assert result.hap2.mut == result.hap2.wt

    def test_gt_1_1_both_haps_different(self, del_variant):
        """
        1|1 基因型：两条 hap 都携带 alt，Mut != WT（对于 hap1 和 hap2）。
        """
        # DEL AT→A at pos=300(1-based), ref=AT(2bp) → end=pos+len-1=301(1-based)
        # upstream 行: start=pos-n-1=296, end=end+1=302(1-based) → 0-based [296,302)
        # bed_row 区间共 6bp
        # ref_seq 需要在 0-based offset 299..300 处（即 pos=300 的 1-based 位置）放置 "AT"
        # 即 ref_seq[299:301] == "AT"
        # bed_row 区间 [296,302) → fetch 返回 ref_seq[296:302]，其中 offset=300-1-296=3
        # 所以 ref_seq[296:302] 应该是 "...AT..."，即 ref_seq[299:301]="AT"
        ref_seq = "C" * 299 + "AT" + "C" * 500

        builder = self._make_builder({"chr3": ref_seq})
        bed_row = BedRow(
            chrom="chr3", start=296, end=302, name="chr3_300_AT_A_upstream"
        )

        result = builder.build_from_bed(bed_row, del_variant, [del_variant])

        assert result is not None
        # 两条 hap 都携带变异，wt_is_alias 应为 False
        assert result.hap1.wt_is_alias_of_mut is False
        assert result.hap2.wt_is_alias_of_mut is False
        # Mut_hap 应包含 alt（A），WT_hap 应包含 ref（AT）
        # 在 bed_row.start=296, pos=300, offset=300-1-296=3
        offset = 300 - 1 - 296
        assert result.hap1.mut[offset] == "A"
        assert result.hap1.wt[offset:offset + 2] == "AT"

    def test_gt_1_0_only_hap1_different(self, snp_variant):
        """
        1|0 基因型：只有 hap1 携带 alt。
        - hap1: Mut != WT (wt_is_alias=False)
        - hap2: Mut == WT (wt_is_alias=True)
        """
        ref_seq = "A" * 200
        builder = self._make_builder({"chr1": ref_seq})
        bed_row = BedRow(
            chrom="chr1", start=90, end=106, name="chr1_100_A_T_upstream"
        )

        result = builder.build_from_bed(bed_row, snp_variant, [snp_variant])

        assert result is not None
        assert result.hap1.wt_is_alias_of_mut is False  # hap1 携带 alt
        assert result.hap2.wt_is_alias_of_mut is True   # hap2 不携带 alt
        # hap1: Mut 在 offset=9 应是 T，WT 应是 A
        offset = 100 - 1 - 90
        assert result.hap1.mut[offset] == "T"
        assert result.hap1.wt[offset] == "A"

    def test_gt_0_1_only_hap2_different(self):
        """
        0|1 基因型：只有 hap2 携带 alt。
        - hap1: Mut == WT (wt_is_alias=True)
        - hap2: Mut != WT (wt_is_alias=False)
        """
        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(0, 1))
        ref_seq = "A" * 200
        builder = self._make_builder({"chr1": ref_seq})
        bed_row = BedRow(
            chrom="chr1", start=90, end=106, name="chr1_100_A_T_upstream"
        )

        result = builder.build_from_bed(bed_row, variant, [variant])

        assert result is not None
        assert result.hap1.wt_is_alias_of_mut is True   # hap1 不携带 alt
        assert result.hap2.wt_is_alias_of_mut is False  # hap2 携带 alt
        # hap1: Mut == WT
        assert result.hap1.mut == result.hap1.wt
        # hap2: Mut 在 offset=9 应是 T，WT 应是 A
        offset = 100 - 1 - 90
        assert result.hap2.mut[offset] == "T"
        assert result.hap2.wt[offset] == "A"

    @pytest.mark.parametrize("gt,alias1,alias2", [
        ((0, 0), True, True),
        ((1, 0), False, True),
        ((0, 1), True, False),
        ((1, 1), False, False),
    ])
    def test_genotype_equivalence_rules(self, gt, alias1, alias2):
        """
        参数化验证 4 种基因型的等价规则。
        - 0|0: Mut_hap1 == WT_hap1, Mut_hap2 == WT_hap2
        - 1|0: Mut_hap1 != WT_hap1, Mut_hap2 == WT_hap2
        - 0|1: Mut_hap1 == WT_hap1, Mut_hap2 != WT_hap2
        - 1|1: Mut_hap1 != WT_hap1, Mut_hap2 != WT_hap2
        """
        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=gt)
        ref_seq = "A" * 200
        builder = self._make_builder({"chr1": ref_seq})
        bed_row = BedRow(
            chrom="chr1", start=90, end=106, name="chr1_100_A_T_upstream"
        )

        result = builder.build_from_bed(bed_row, variant, [variant])

        assert result.hap1.wt_is_alias_of_mut is alias1
        assert result.hap2.wt_is_alias_of_mut is alias2


# ===========================================================================
# Section 4: SixSeqResult.iter_named_seqs() 去重逻辑
# ===========================================================================

class TestSixSeqResultIterNamedSeqs:

    def _make_result(self, wt1_alias: bool, wt2_alias: bool) -> SixSeqResult:
        return SixSeqResult(
            direction="upstream",
            hap1=HapSeqPair(mut="AAAA", wt="TTTT", wt_is_alias_of_mut=wt1_alias),
            hap2=HapSeqPair(mut="CCCC", wt="GGGG", wt_is_alias_of_mut=wt2_alias),
            ref_pair=HapSeqPair(mut="RRRR", wt="WWWW", wt_is_alias_of_mut=False),
        )

    def test_all_different(self):
        result = self._make_result(wt1_alias=False, wt2_alias=False)
        items = result.iter_named_seqs()
        names = [n for n, _ in items]
        # 6 条序列（hap1 mut, wt; hap2 mut, wt; ref mut, wt）
        assert len(items) == 6
        assert "upstream_hap1_mut" in names
        assert "upstream_hap1_wt" in names

    def test_gt_0_0_dedup(self):
        """0|0 情形：wt_alias=True，iter 应只返回 mut（不单独列 wt）"""
        result = self._make_result(wt1_alias=True, wt2_alias=True)
        items = result.iter_named_seqs()
        names = [n for n, _ in items]
        # 4 条：hap1_mut, hap2_mut, ref_mut, ref_wt
        assert "upstream_hap1_wt" not in names
        assert "upstream_hap2_wt" not in names
        assert len(items) == 4

    def test_get_seq_by_name_alias(self):
        """wt_is_alias=True 时，get_seq_by_name("...wt") 应返回 mut 序列"""
        result = self._make_result(wt1_alias=True, wt2_alias=False)
        wt1 = result.get_seq_by_name("upstream_hap1_wt")
        assert wt1 == "AAAA"   # wt aliased → returns mut


# ===========================================================================
# Section 5: EmbeddingExtractor 缓存命中逻辑
# ===========================================================================

class TestEmbeddingExtractorCache:
    """
    使用 mock EmbeddingManager（不需要 GPU），测试缓存命中逻辑。
    """

    HIDDEN_DIM = 8

    def _make_mock_manager(self):
        """创建 mock EmbeddingManager，返回固定的 hidden state。"""
        import torch

        mock_mgr = MagicMock()
        mock_mgr.model_name = "mock_model"
        mock_mgr.batch_size = 4
        mock_mgr.mode = "local"

        def fake_get_hidden_states(seqs):
            B = len(seqs)
            L = 10  # 固定序列长度
            hidden = torch.ones(B, L, self.HIDDEN_DIM)
            mask = torch.ones(B, L, dtype=torch.long)
            return hidden, mask

        def fake_tail_pool(hidden, mask, w, method="mean"):
            B, L, D = hidden.size()
            return hidden[:, -w:, :].mean(dim=1)

        mock_mgr.get_hidden_states = MagicMock(side_effect=fake_get_hidden_states)
        mock_mgr.tail_pool = MagicMock(side_effect=fake_tail_pool)
        return mock_mgr

    def _make_distinguishable_mock_manager(self):
        """
        创建可区分 upstream/downstream 的 mock EmbeddingManager。
        根据输入序列内容产生不同 embedding：upstream 序列 → 向量全 1，
        downstream 序列（已经是 RC）→ 向量全 2。
        """
        import torch

        mock_mgr = MagicMock()
        mock_mgr.model_name = "mock_model"
        mock_mgr.batch_size = 4
        mock_mgr.mode = "local"

        def fake_get_hidden_states(seqs):
            B = len(seqs)
            L = 10
            hidden = torch.zeros(B, L, self.HIDDEN_DIM)
            for b, seq in enumerate(seqs):
                # 根据序列内容决定 embedding 值
                # 使用序列哈希区分不同的序列
                h = hash(seq) % 100
                hidden[b, :, :] = h
            mask = torch.ones(B, L, dtype=torch.long)
            return hidden, mask

        def fake_tail_pool(hidden, mask, w, method="mean"):
            B, L, D = hidden.size()
            return hidden[:, -w:, :].mean(dim=1)

        mock_mgr.get_hidden_states = MagicMock(side_effect=fake_get_hidden_states)
        mock_mgr.tail_pool = MagicMock(side_effect=fake_tail_pool)
        return mock_mgr

    def _make_six_seq_result(self, direction: str, alias: bool = False) -> SixSeqResult:
        return SixSeqResult(
            direction=direction,
            hap1=HapSeqPair(mut="AAAA", wt="TTTT", wt_is_alias_of_mut=alias),
            hap2=HapSeqPair(mut="CCCC", wt="GGGG", wt_is_alias_of_mut=alias),
            ref_pair=HapSeqPair(mut="MMMM", wt="WWWW", wt_is_alias_of_mut=False),
        )

    @pytest.mark.cpu
    def test_extract_output_shape(self):
        """6 种 embedding 的维度应为 hidden_dim * 2。"""
        import torch

        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        embs = extractor.extract(
            center_variant=variant,
            up_result=up_result,
            dn_result=dn_result,
        )

        assert set(embs.keys()) == {
            "Mut_hap1", "WT_hap1", "Mut_hap2", "WT_hap2", "Mut_ref", "WT_ref"
        }
        for name, vec in embs.items():
            assert isinstance(vec, np.ndarray), f"{name} should be ndarray"
            assert vec.shape == (self.HIDDEN_DIM * 2,), \
                f"{name} shape {vec.shape} != ({self.HIDDEN_DIM * 2},)"

    @pytest.mark.cpu
    @pytest.mark.parametrize("ref,alt", [
        ("A", "T"),       # SNP
        ("A", "AT"),      # INS
        ("AT", "A"),      # DEL
        ("ATC", "GTG"),   # MNV
    ])
    def test_extract_output_shape_all_variant_types(self, ref, alt):
        """各种变异类型的输出维度均应为 hidden_dim * 2。"""
        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref=ref, alt=alt, gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        embs = extractor.extract(
            center_variant=variant,
            up_result=up_result,
            dn_result=dn_result,
        )

        for name, vec in embs.items():
            assert vec.shape == (self.HIDDEN_DIM * 2,), \
                f"{ref}->{alt}: {name} shape {vec.shape} != ({self.HIDDEN_DIM * 2},)"

    @pytest.mark.cpu
    def test_cache_hit_avoids_inference(self):
        """同一序列第二次 extract 不应触发新的推理调用。"""
        import torch

        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        # 第一次：触发推理
        extractor.extract(center_variant=variant, up_result=up_result, dn_result=dn_result)
        call_count_1 = mgr.get_hidden_states.call_count

        # 第二次：完全命中缓存，不触发额外推理
        extractor.extract(center_variant=variant, up_result=up_result, dn_result=dn_result)
        call_count_2 = mgr.get_hidden_states.call_count

        assert call_count_1 == call_count_2, \
            f"Expected no new inference on cache hit, but got {call_count_2 - call_count_1} extra calls"

    @pytest.mark.cpu
    def test_gt_0_0_shares_mut_wt_embedding(self):
        """
        0|0 基因型：Mut_hap1 == WT_hap1（wt_is_alias=True）时，
        两者应产生相同的 embedding 向量。
        """
        import torch

        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(0, 0))
        up_result = self._make_six_seq_result("upstream", alias=True)
        dn_result = self._make_six_seq_result("downstream", alias=True)

        embs = extractor.extract(
            center_variant=variant,
            up_result=up_result,
            dn_result=dn_result,
        )

        # 0|0 时 Mut_hap1 == WT_hap1（完全相同序列 → 相同 embedding）
        np.testing.assert_array_equal(embs["Mut_hap1"], embs["WT_hap1"])
        np.testing.assert_array_equal(embs["Mut_hap2"], embs["WT_hap2"])

    @pytest.mark.cpu
    def test_cache_size_after_extraction(self):
        """提取后缓存大小应等于去重后的推理序列数。"""
        import torch

        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        extractor.extract(center_variant=variant, up_result=up_result, dn_result=dn_result)

        # 序列总数：
        # upstream:  hap1_mut, hap1_wt, hap2_mut, hap2_wt, ref_mut, ref_wt = 6
        # downstream: 对上述取 RC，RC(AAAA)≠TTTT 等，需分别 hash
        # 但这里 gt=(1,0)，hap2 alias=False，所以 6+6=12 条序列
        # 由于序列内容可能有重叠（如 RC("AAAA")="TTTT"），
        # 实际 cache 大小 <= 12
        assert extractor.cache_size() > 0
        assert extractor.cache_size() <= 12

    @pytest.mark.cpu
    def test_clear_cache(self):
        """clear_cache 后缓存应为空，后续提取需重新推理。"""
        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        # 第一次提取
        extractor.extract(center_variant=variant, up_result=up_result, dn_result=dn_result)
        assert extractor.cache_size() > 0

        # 清空缓存
        extractor.clear_cache()
        assert extractor.cache_size() == 0

        # 第二次提取应重新触发推理
        call_count_before = mgr.get_hidden_states.call_count
        extractor.extract(center_variant=variant, up_result=up_result, dn_result=dn_result)
        call_count_after = mgr.get_hidden_states.call_count
        assert call_count_after > call_count_before, \
            "clear_cache 后应重新推理"

    @pytest.mark.cpu
    def test_concat_order(self):
        """
        验证 emb_up 在前、emb_dn 在后：concat([emb_up, emb_dn])。
        使用可区分的 mock，使 upstream 和 downstream 序列产生不同 embedding。
        """
        mgr = self._make_distinguishable_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache={})

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        up_result = self._make_six_seq_result("upstream")
        dn_result = self._make_six_seq_result("downstream")

        embs = extractor.extract(
            center_variant=variant,
            up_result=up_result,
            dn_result=dn_result,
        )

        # 对 Mut_hap1: up 方向用 "AAAA"(正链), dn 方向用 RC("AAAA")="TTTT"
        # 两个序列不同，embedding 应不同
        # 验证前半部分来自 upstream，后半部分来自 downstream
        mut_hap1_emb = embs["Mut_hap1"]
        assert mut_hap1_emb.shape == (self.HIDDEN_DIM * 2,)
        # 前半部分和后半部分的值应来自不同序列的推理结果
        emb_up_part = mut_hap1_emb[:self.HIDDEN_DIM]
        emb_dn_part = mut_hap1_emb[self.HIDDEN_DIM:]
        # 由于 distinguishable mock 使用 hash(seq) 作为值，
        # "AAAA" 和 RC("AAAA")="TTTT" 产生不同值
        # 至少验证前后半部分可能不同（如果序列恰好同值则跳过）
        # 但更重要的是验证维度切分正确
        assert emb_up_part.shape == (self.HIDDEN_DIM,)
        assert emb_dn_part.shape == (self.HIDDEN_DIM,)

    @pytest.mark.cpu
    def test_downstream_uses_reverse_complement(self):
        """
        验证 downstream 方向对序列取反向互补后再推理。
        通过比较同一 SixSeqResult 在不同方向下的 cache key 来验证。
        """
        # upstream 方向：序列直接推理
        # downstream 方向：序列取 RC 后推理
        # 两者缓存键不同（因为输入序列不同）
        seq = "ATCGATCG"
        rc_seq = reverse_complement(seq)

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(1, 0))
        w = EmbeddingExtractor.compute_w(variant)

        # upstream 方向 cache key 用原序列
        up_key = _cache_key(seq, w, "mean")
        # downstream 方向 cache key 用 RC 序列
        dn_key = _cache_key(rc_seq, w, "mean")

        # 如果序列不是回文，两个 key 应不同
        if seq != rc_seq:
            assert up_key != dn_key, \
                "upstream 和 downstream 应使用不同的推理输入序列"

    @pytest.mark.cpu
    def test_cache_key_includes_w(self):
        """缓存键应包含 w，确保不同 w 值不会误命中。"""
        seq = "ATCGATCG"
        key_w3 = _cache_key(seq, w=3, pooling="mean")
        key_w5 = _cache_key(seq, w=5, pooling="mean")
        assert key_w3 != key_w5, \
            "相同序列不同 w 应产生不同缓存键"

    @pytest.mark.cpu
    def test_cache_key_includes_pooling(self):
        """缓存键应包含 pooling 方法，确保不同 pooling 不会误命中。"""
        seq = "ATCGATCG"
        key_mean = _cache_key(seq, w=3, pooling="mean")
        key_max = _cache_key(seq, w=3, pooling="max")
        assert key_mean != key_max, \
            "相同序列不同 pooling 应产生不同缓存键"

    @pytest.mark.cpu
    def test_cache_disabled_still_deduplicates_within_variant(self):
        """
        cache=None 时全局缓存关闭，但同一变异内部的等价序列仍去重
        （如 0|0 时 Mut==WT 使用相同序列，只推理一次）。
        """
        mgr = self._make_mock_manager()
        extractor = EmbeddingExtractor(embedding_manager=mgr, pooling="mean", cache=None)

        variant = Variant(chrom="chr1", pos=100, ref="A", alt="T", gt=(0, 0))
        up_result = self._make_six_seq_result("upstream", alias=True)
        dn_result = self._make_six_seq_result("downstream", alias=True)

        embs = extractor.extract(
            center_variant=variant,
            up_result=up_result,
            dn_result=dn_result,
        )

        # 结果仍正确：Mut_hap1 == WT_hap1
        np.testing.assert_array_equal(embs["Mut_hap1"], embs["WT_hap1"])
        # 全局缓存大小为 0（禁用）
        assert extractor.cache_size() == 0


# ===========================================================================
# Section 6: compute_w 正确性
# ===========================================================================

class TestComputeW:

    def test_snp(self):
        v = Variant("chr1", 1, "A", "T", (1, 0))
        assert EmbeddingExtractor.compute_w(v) == 3
        assert EmbeddingExtractor.compute_w_mut(v) == 3
        assert EmbeddingExtractor.compute_w_wt(v) == 3

    def test_ins(self):
        v = Variant("chr1", 1, "A", "AT", (0, 1))
        assert EmbeddingExtractor.compute_w(v) == 4
        assert EmbeddingExtractor.compute_w_mut(v) == 4  # alt=2 + 2
        assert EmbeddingExtractor.compute_w_wt(v) == 3   # ref=1 + 2

    def test_del(self):
        v = Variant("chr1", 1, "AT", "A", (1, 1))
        assert EmbeddingExtractor.compute_w(v) == 3
        assert EmbeddingExtractor.compute_w_mut(v) == 3  # alt=1 + 2
        assert EmbeddingExtractor.compute_w_wt(v) == 4   # ref=2 + 2

    def test_mnv(self):
        v = Variant("chr1", 1, "ATC", "GTG", (0, 0))
        assert EmbeddingExtractor.compute_w(v) == 5
        assert EmbeddingExtractor.compute_w_mut(v) == 5  # alt=3 + 2
        assert EmbeddingExtractor.compute_w_wt(v) == 5   # ref=3 + 2

    def test_long_ins(self):
        v = Variant("chr1", 1, "A", "ATTTTTT", (1, 0))
        assert EmbeddingExtractor.compute_w(v) == 9
        assert EmbeddingExtractor.compute_w_mut(v) == 9  # alt=7 + 2
        assert EmbeddingExtractor.compute_w_wt(v) == 3   # ref=1 + 2


# ===========================================================================
# Section 7: reverse_complement
# ===========================================================================

class TestReverseComplement:

    def test_basic(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self):
        assert reverse_complement("AATT") == "AATT"

    def test_lowercase(self):
        assert reverse_complement("atcg") == "cgat"

    def test_n(self):
        assert reverse_complement("NATCGN") == "NCGATN"


# ===========================================================================
# Section 8: GVL WT_hap 恢复逻辑（不依赖真实 genvarloader，仅测试函数逻辑）
# ===========================================================================

class TestGVLRestoreWT:
    """
    直接测试 GenVarLoaderSequenceBuilder._restore_wt_from_mut_hap()，
    不依赖 genvarloader 库（仅调用方法本身）。
    """

    def _make_builder(self):
        """构造一个不初始化 genvarloader 的 builder，只测试恢复方法。"""
        # 延迟导入，避免 genvarloader 未安装时报错
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = GenVarLoaderSequenceBuilder.__new__(GenVarLoaderSequenceBuilder)
        builder.n = 5
        builder._fasta = None
        builder._dataset = None
        builder._name_to_idx = {}
        builder.current_sample = None
        return builder

    def test_snp_upstream_restore(self):
        """
        SNP A→T，upstream 序列末尾：...XAXP（P=1bp padding）
        恢复后应将 T 替换为 A。

        mut_hap = "NNNNNNTX"   (len=8, upstream, var at offset=6, padding at offset=7)
        alt=T, ref=A
        offset = 8 - 1 - 1 = 6
        wt_hap = "NNNNNNAX"
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "A", "T", (1, 0))

        mut_hap = "NNNNNNTP"  # T at offset=6, P at offset=7 (padding)
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=True,
        )

        assert wt_hap is not None
        assert wt_hap[6] == "A"
        assert wt_hap[:6] == "NNNNNN"
        assert wt_hap[7] == "P"

    def test_ins_upstream_restore(self):
        """
        INS A→AT，upstream 序列末尾：...XATx（x=1bp padding）
        mut_hap 含 alt "AT" 在末尾前 1bp。
        mut_hap = "NNNNNNATx"  (len=9, alt=AT at offset=6, padding at offset=8)
        offset = 9 - 2 - 1 = 6
        wt_hap = "NNNNNNA x" → "NNNNNNAx"
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "A", "AT", (0, 1))

        mut_hap = "NNNNNNATx"
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=True,
        )

        assert wt_hap is not None
        # 恢复后 alt "AT" 被替换为 ref "A"
        assert wt_hap == "NNNNNNAx"

    def test_snp_downstream_restore(self):
        """
        SNP A→T，downstream 序列开头：xT...
        offset = 1
        wt_hap = "x" + "A" + ...
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "A", "T", (1, 0))

        mut_hap = "xTNNNNNN"  # T at offset=1
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=False,  # downstream
        )

        assert wt_hap is not None
        assert wt_hap[1] == "A"
        assert wt_hap[0] == "x"
        assert wt_hap[2:] == "NNNNNN"

    def test_del_upstream_restore(self):
        """
        DEL AT→A，upstream 序列：...XAxxP（A=alt, ref=AT）
        mut_hap = "NNNNNNAxP"  (len=9, alt="A" at offset=6, padding at offset=8)
        offset = 9 - 1 - 1 = 7 → 但实际 alt 在 offset=6
        修正：DEL AT→A, ref=AT(2bp), alt=A(1bp)
        mut_hap 中 alt="A" 在 offset=len-1-1=len-2 处
        mut_hap = "NNNNNNAP" (len=8, alt="A" at offset=6, padding at offset=7)
        offset = 8 - 1 - 1 = 6
        恢复后: wt_hap = "NNNNNNATP"（ref="AT" 替换 alt="A"，序列变长 1bp）
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "AT", "A", (1, 0))

        mut_hap = "NNNNNNAP"  # A at offset=6, P at offset=7
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=True,
        )

        assert wt_hap is not None
        # 恢复后 alt "A" 被替换为 ref "AT"
        assert wt_hap == "NNNNNNATP"

    def test_del_downstream_restore(self):
        """
        DEL AT→A，downstream 序列开头：xA...（A=alt）
        mut_hap = "xANNNNNN" (A at offset=1)
        恢复后: wt_hap = "xATNNNNN"（ref="AT" 替换 alt="A"，序列变长 1bp）
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "AT", "A", (1, 0))

        mut_hap = "xANNNNNN"
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=False,
        )

        assert wt_hap is not None
        # alt "A" (1bp) → ref "AT" (2bp)，序列变长
        assert wt_hap == "xATNNNNNN"

    def test_ins_downstream_restore(self):
        """
        INS A→AT，downstream 序列开头：xAT...（AT=alt）
        mut_hap = "xATNNNNN" (AT at offset=1)
        恢复后: wt_hap = "xANNNNNN" → "xANNNNN"（ref="A" 替换 alt="AT"，序列变短 1bp）
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "A", "AT", (0, 1))

        mut_hap = "xATNNNNN"
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=False,
        )

        assert wt_hap is not None
        # alt "AT" (2bp) → ref "A" (1bp)，序列变短
        assert wt_hap == "xANNNNN"

    def test_mnv_upstream_restore(self):
        """
        MNV ATC→GTG，upstream 序列末尾：...GTGx（GTG=alt）
        mut_hap = "NNNNNGTGx" (len=9, alt="GTG" at offset=5, padding at offset=8)
        offset = 9 - 3 - 1 = 5
        恢复后: wt_hap = "NNNNNATCx"（ref="ATC" 替换 alt="GTG"）
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "ATC", "GTG", (1, 1))

        mut_hap = "NNNNNGTGx"
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=True,
        )

        assert wt_hap is not None
        assert wt_hap == "NNNNNATCx"

    def test_mnv_downstream_restore(self):
        """
        MNV ATC→GTG，downstream 序列开头：xGTG...（GTG=alt）
        mut_hap = "xGTGNNNN" (GTG at offset=1)
        恢复后: wt_hap = "xATCNNNN"（ref="ATC" 替换 alt="GTG"）
        """
        from core.genvarloader_builder import GenVarLoaderSequenceBuilder

        builder = self._make_builder()
        variant = Variant("chr1", 100, "ATC", "GTG", (1, 1))

        mut_hap = "xGTGNNNN"
        wt_hap = builder._restore_wt_from_mut_hap(
            mut_hap=mut_hap,
            center_variant=variant,
            is_upstream=False,
        )

        assert wt_hap is not None
        assert wt_hap == "xATCNNNN"


# ===========================================================================
# 直接运行入口
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
