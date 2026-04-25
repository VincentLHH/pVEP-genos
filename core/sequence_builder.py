"""
core/sequence_builder.py
========================
本地（builtin）序列构建器，基于 pysam 直接从参考基因组重建 6 种单倍型序列。

新版 6 种序列定义
-----------------
Mut_hap1 : 真实样本单倍型1背景 + 目标变异（若 hap1 携带 alt 则保留，否则等同 WT_hap1）
WT_hap1  : 真实样本单倍型1背景 + 目标位点强制恢复为 ref（即从 Mut_hap1 去掉目标变异）
Mut_hap2 : 同上，针对单倍型2
WT_hap2  : 同上，针对单倍型2
Mut_ref  : 参考基因组背景 + 强制插入目标变异
WT_ref   : 纯参考基因组序列（无任何变异）

基因型等价规则
--------------
0|0 : Mut_hap1 == WT_hap1, Mut_hap2 == WT_hap2
1|0 : Mut_hap1 != WT_hap1, Mut_hap2 == WT_hap2
0|1 : Mut_hap1 == WT_hap1, Mut_hap2 != WT_hap2
1|1 : Mut_hap1 != WT_hap1, Mut_hap2 != WT_hap2

BED 拆分模式
------------
输入 BedRow 有两种方向：upstream 和 downstream。
对于 upstream 行，构建正链序列（变异在末尾 w 个 token 附近）；
对于 downstream 行，需要反向互补后使用（变异同样在末尾）。

使用示例
--------
    builder = SequenceBuilder(fasta_path, window_size=128)
    bed_rows = load_bed("splits.bed")  # 已拆分的 BED
    for row in bed_rows:
        result = builder.build_from_bed(row, center_variant, all_variants_in_region)
        # 返回 SixSeqResult
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pysam

from core.variant import BedRow, Variant, VariantBedSplit, split_variant_to_bed


# ===========================================================================
# 反向互补
# ===========================================================================

_COMP_TABLE = str.maketrans("ATCGNatcgn", "TAGCNtagcn")


def reverse_complement(seq: str) -> str:
    return seq[::-1].translate(_COMP_TABLE)


# ===========================================================================
# 结果数据类
# ===========================================================================

@dataclass
class HapSeqPair:
    """
    一种单倍型（hap1/hap2/ref）对应的 Mut/WT 序列对。

    注意：同一个 BedRow 方向（upstream/downstream）下的序列。
    downstream 序列存储的是正链，推理前需调用 reverse_complement()。
    """
    mut: str    # Mut 序列（正链，upstream方向）
    wt: str     # WT  序列（正链，upstream方向）
    wt_is_alias_of_mut: bool = False  # 若为 True，wt 与 mut 指向同一序列（基因型等价）


@dataclass
class SixSeqResult:
    """
    一个变异-BedRow 对应的全部 6 种序列（含方向信息）。

    字段：
        direction : "upstream" 或 "downstream"
        hap1      : HapSeqPair（Mut_hap1 / WT_hap1）
        hap2      : HapSeqPair（Mut_hap2 / WT_hap2）
        ref_pair  : HapSeqPair（Mut_ref  / WT_ref，wt_is_alias_of_mut=False）

    下游使用时：
        upstream   → 正链输入模型
        downstream → 先 reverse_complement() 再输入
    """
    direction: str  # "upstream" or "downstream"
    hap1: HapSeqPair
    hap2: HapSeqPair
    ref_pair: HapSeqPair   # WT_ref / Mut_ref

    def iter_named_seqs(self) -> List[Tuple[str, str]]:
        """
        迭代所有 (名称, 序列) 对，用于批推理的 flat_seq_dict 构建。
        名称格式："{direction}_{hap_type}_{mut_or_wt}"
        已在基因型等价规则下去重（wt_is_alias_of_mut 时 WT 不单独出现）。
        """
        items: List[Tuple[str, str]] = []
        d = self.direction
        for hap_name, pair in [("hap1", self.hap1), ("hap2", self.hap2), ("ref", self.ref_pair)]:
            items.append((f"{d}_{hap_name}_mut", pair.mut))
            if not pair.wt_is_alias_of_mut:
                items.append((f"{d}_{hap_name}_wt", pair.wt))
        return items

    def get_seq_by_name(self, name: str) -> Optional[str]:
        """
        按名称（如 'upstream_hap1_wt'）获取序列；
        若等价规则导致 wt == mut，直接返回 mut 序列。
        """
        parts = name.split("_", 2)
        if len(parts) != 3:
            return None
        _dir, hap_name, mut_or_wt = parts
        pair_map = {"hap1": self.hap1, "hap2": self.hap2, "ref": self.ref_pair}
        pair = pair_map.get(hap_name)
        if pair is None:
            return None
        if mut_or_wt == "mut":
            return pair.mut
        elif mut_or_wt == "wt":
            return pair.wt if not pair.wt_is_alias_of_mut else pair.mut
        return None


# ===========================================================================
# 主类
# ===========================================================================

class SequenceBuilder:
    """
    本地序列构建器。

    主要接口
    --------
    build_six_seqs(bed_split, center_variant, variants_in_region)
        → {"upstream": SixSeqResult, "downstream": SixSeqResult}

    build_from_bed(bed_row, center_variant, variants_in_region)
        → SixSeqResult（单行）

    旧接口（已废弃，保留兼容）
    --------
    build(center_variant, variants_in_region)
    """

    def __init__(
        self,
        fasta_path: str,
        window_size: int = 128,
        n: int = 200,
    ):
        """
        参数
        ----
        fasta_path   : 参考基因组 FASTA 文件路径
        window_size  : 模型推理上下文窗口（bp），用于校验 n
        n            : BED 拆分时的侧翼扩展长度（bp）
        """
        self.fasta = pysam.FastaFile(fasta_path)
        self.window_size = window_size
        self.half_window = window_size // 2
        self.n = n

    # -----------------------------------------------------------------------
    # 🔥 新主接口：一次性构建上游 + 下游两行的 6 种序列
    # -----------------------------------------------------------------------

    def build_six_seqs(
        self,
        bed_split: VariantBedSplit,
        center_variant: Variant,
        variants_in_region: List[Variant],
    ) -> Optional[Dict[str, SixSeqResult]]:
        """
        对 VariantBedSplit 中的上游和下游 BedRow 各构建 SixSeqResult。

        返回
        ----
        {"upstream": SixSeqResult, "downstream": SixSeqResult}
        若任意一行构建失败则返回 None。
        """
        up_result = self.build_from_bed(
            bed_split.upstream, center_variant, variants_in_region
        )
        if up_result is None:
            return None

        dn_result = self.build_from_bed(
            bed_split.downstream, center_variant, variants_in_region
        )
        if dn_result is None:
            return None

        return {"upstream": up_result, "downstream": dn_result}

    def build_from_bed(
        self,
        bed_row: BedRow,
        center_variant: Variant,
        variants_in_region: List[Variant],
    ) -> Optional[SixSeqResult]:
        """
        基于一行 BedRow 构建 SixSeqResult（6 种序列）。

        步骤
        ----
        1. 从 FASTA 获取 WT_ref（即 BedRow 对应区域的纯参考序列）
        2. 根据 center_variant 构建 Mut_ref（在 WT_ref 中强制插入目标变异）
        3. 从 FASTA 获取背景序列，应用区域内所有背景变异，得到 hap1/hap2 基础序列
        4. 根据基因型决定 Mut_hap/WT_hap：
           - 若该 hap 携带 alt → Mut_hap = 真实 hap（含目标变异），WT_hap = 从 Mut_hap 中移除目标变异
           - 若该 hap 不携带 alt → Mut_hap = WT_hap = 真实 hap（无目标变异）
        """
        # 1. 从 FASTA 获取 WT_ref
        wt_ref = self._fetch_bed_row(bed_row)
        if wt_ref is None:
            return None

        # 2. 构建 Mut_ref（在 WT_ref 的 center_variant 位置强制插入 alt）
        mut_ref = self._inject_variant_into_ref(wt_ref, bed_row, center_variant)
        if mut_ref is None:
            return None

        # 3. 过滤区域内所有变异（包含目标变异和背景变异）
        region_vars = self._filter_variants(variants_in_region, bed_row)

        # 4. 分别构建 hap1 和 hap2
        hap1_pair = self._build_hap_pair(
            wt_ref=wt_ref,
            bed_row=bed_row,
            center_variant=center_variant,
            variants_in_region=region_vars,
            hap_index=0,
        )
        hap2_pair = self._build_hap_pair(
            wt_ref=wt_ref,
            bed_row=bed_row,
            center_variant=center_variant,
            variants_in_region=region_vars,
            hap_index=1,
        )

        direction = "upstream" if bed_row.is_upstream else "downstream"

        return SixSeqResult(
            direction=direction,
            hap1=hap1_pair,
            hap2=hap2_pair,
            ref_pair=HapSeqPair(
                mut=mut_ref,
                wt=wt_ref,
                wt_is_alias_of_mut=False,
            ),
        )

    # -----------------------------------------------------------------------
    # 内部实现
    # -----------------------------------------------------------------------

    def _fetch_bed_row(self, bed_row: BedRow) -> Optional[str]:
        """
        从 FASTA 按 BedRow 的 0-based 坐标提取序列（即 WT_ref）。
        pysam.fetch 接受 0-based 左闭右开坐标。
        """
        try:
            seq = self.fasta.fetch(
                bed_row.chrom, bed_row.start, bed_row.end
            ).upper()
            return seq
        except Exception:
            return None

    def _inject_variant_into_ref(
        self,
        ref_seq: str,
        bed_row: BedRow,
        variant: Variant,
    ) -> Optional[str]:
        """
        在 ref_seq 中把目标变异的 ref 等位基因替换为 alt（强制插入，不考虑样本基因型）。

        坐标换算
        --------
        variant.pos 是 1-based，bed_row.start 是 0-based。
        变异在序列中的偏移 = variant.pos - 1 - bed_row.start
        """
        offset = variant.pos - 1 - bed_row.start  # 0-based 偏移

        if offset < 0 or offset + len(variant.ref) > len(ref_seq):
            return None

        current = ref_seq[offset: offset + len(variant.ref)]
        if current != variant.ref:
            # REF 不匹配（可能是软掩码或坐标错误），依然尝试注入但打印警告
            import warnings
            warnings.warn(
                f"Variant {variant.id}: ref mismatch at offset {offset}. "
                f"Expected {variant.ref!r}, found {current!r}. Forcing injection."
            )

        mut_seq = ref_seq[:offset] + variant.alt + ref_seq[offset + len(variant.ref):]
        return mut_seq

    def _filter_variants(
        self,
        variants: List[Variant],
        bed_row: BedRow,
    ) -> List[Variant]:
        """
        过滤出位置落在 bed_row 范围内的变异。
        bed_row 坐标 0-based；variant.pos 是 1-based。
        变异位置转为 0-based：variant.pos - 1
        """
        result = []
        for v in variants:
            v0 = v.pos - 1  # 0-based
            if bed_row.start <= v0 < bed_row.end and v.gt is not None:
                result.append(v)
        return result

    def _build_hap_pair(
        self,
        wt_ref: str,
        bed_row: BedRow,
        center_variant: Variant,
        variants_in_region: List[Variant],
        hap_index: int,
    ) -> HapSeqPair:
        """
        构建指定单倍型（hap_index=0 → hap1，1 → hap2）的 Mut/WT 序列对。

        逻辑
        ----
        1. 应用除 center_variant 以外的所有背景变异（该 hap 携带的） → 得到背景序列 bg_seq
        2. 判断该 hap 是否携带 center_variant 的 alt：
           - 携带（allele=1）：
               Mut_hap = bg_seq 再插入 center_variant.alt
               WT_hap  = bg_seq（即已应用背景但不含目标变异）
           - 不携带（allele=0）：
               Mut_hap = WT_hap = bg_seq（完全等价，设 wt_is_alias_of_mut=True）
        """
        # 背景变异：区域内除 center_variant 以外的所有变异（该单倍型携带的）
        bg_vars = [
            v for v in variants_in_region
            if v.id != center_variant.id
        ]

        # 应用背景变异，得到背景序列
        bg_seq = self._apply_variants(wt_ref, bed_row.start, bg_vars, hap_index)

        # 判断该 hap 是否携带目标变异
        hap_has_alt = (
            center_variant.gt is not None
            and center_variant.gt[hap_index] == 1
        )

        if hap_has_alt:
            # Mut_hap = 在背景序列上再插入目标变异
            # 注意：bg_seq 因 indel 背景变异，长度可能已变化，
            # 需要追踪 center_variant 在 bg_seq 中的真实偏移
            mut_hap = self._inject_center_into_bg(
                bg_seq=bg_seq,
                bg_vars=bg_vars,
                hap_index=hap_index,
                bed_start=bed_row.start,
                center_variant=center_variant,
            )
            if mut_hap is None:
                # 注入失败，回退：Mut = WT = bg_seq
                return HapSeqPair(mut=bg_seq, wt=bg_seq, wt_is_alias_of_mut=True)
            return HapSeqPair(mut=mut_hap, wt=bg_seq, wt_is_alias_of_mut=False)
        else:
            # 该 hap 不携带目标变异，Mut == WT
            return HapSeqPair(mut=bg_seq, wt=bg_seq, wt_is_alias_of_mut=True)

    def _apply_variants(
        self,
        ref_seq: str,
        seq_start: int,  # 0-based，序列起始位置（= bed_row.start）
        variants: List[Variant],
        hap_index: int,
    ) -> str:
        """
        在 ref_seq 中按顺序应用 variants 中该 hap 携带的变异。

        参数
        ----
        ref_seq   : 初始序列（来自参考基因组，0-based 起点 = seq_start）
        seq_start : ref_seq 对应的 0-based 起始位置（与 variant.pos-1 对齐）
        variants  : 要应用的变异列表（含 gt）
        hap_index : 0 → hap1, 1 → hap2

        Returns
        -------
        应用变异后的字符串序列
        """
        seq = ref_seq
        shift = 0  # 累积因 indel 导致的序列长度偏移

        for v in sorted(variants, key=lambda x: x.pos):
            if v.gt is None or v.gt[hap_index] == 0:
                continue

            # 0-based 位置偏移
            pos_in_seq = (v.pos - 1 - seq_start) + shift

            if pos_in_seq < 0 or pos_in_seq + len(v.ref) > len(seq):
                continue

            current = seq[pos_in_seq: pos_in_seq + len(v.ref)]
            if current.upper() != v.ref.upper():
                continue

            seq = seq[:pos_in_seq] + v.alt + seq[pos_in_seq + len(v.ref):]
            shift += len(v.alt) - len(v.ref)

        return seq

    def _inject_center_into_bg(
        self,
        bg_seq: str,
        bg_vars: List[Variant],
        hap_index: int,
        bed_start: int,
        center_variant: Variant,
    ) -> Optional[str]:
        """
        将 center_variant（alt）注入到已应用背景变异的序列 bg_seq 中。

        关键：bg_seq 已因背景 indel 变化了长度，center_variant 的位置
        需要根据背景变异的位移（shift）动态计算。

        Returns
        -------
        注入后的字符串，或 None（若偏移越界）
        """
        # 计算 center_variant 在 bg_seq 中的真实偏移
        shift = 0
        for v in sorted(bg_vars, key=lambda x: x.pos):
            if v.gt is None or v.gt[hap_index] == 0:
                continue
            if v.pos < center_variant.pos:
                shift += len(v.alt) - len(v.ref)

        center_offset = (center_variant.pos - 1 - bed_start) + shift

        if center_offset < 0 or center_offset + len(center_variant.ref) > len(bg_seq):
            return None

        # 注入 alt（不校验 ref，因为背景可能已修改序列）
        return (
            bg_seq[:center_offset]
            + center_variant.alt
            + bg_seq[center_offset + len(center_variant.ref):]
        )

    def reverse_complement(self, seq: str) -> str:
        """返回序列的反向互补（实例方法，与旧版 API 兼容）。"""
        return reverse_complement(seq)

    # -----------------------------------------------------------------------
    # 旧版接口（已废弃，保留兼容）
    # -----------------------------------------------------------------------

    def build(
        self,
        center_variant: Variant,
        variants_in_region: List[Variant],
    ) -> Optional[Dict]:
        """
        [已废弃] 旧版构建接口，仅供兼容旧测试使用。

        新代码请使用 build_six_seqs() 或 build_from_bed()。
        """
        import warnings
        warnings.warn(
            "SequenceBuilder.build() is deprecated. "
            "Use build_six_seqs() or build_from_bed() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # 构建旧格式：以 center_variant 为中心的 window_size 窗口
        pos = center_variant.pos
        left = pos - self.half_window
        right = pos + len(center_variant.ref) - 1 + self.half_window

        if left < 1:
            return None

        try:
            ref_seq = self.fasta.fetch(
                center_variant.chrom, left - 1, right
            ).upper()
        except Exception:
            return None

        # 过滤窗口内变异
        region_vars = [
            v for v in variants_in_region
            if left <= v.pos <= right and v.gt is not None
        ]

        # 构建 hap1 / hap2（旧方法：直接应用所有变异）
        hap1_seq = self._apply_variants(ref_seq, left - 1, region_vars, hap_index=0)
        hap2_seq = self._apply_variants(ref_seq, left - 1, region_vars, hap_index=1)

        return {
            "ref_seq": ref_seq,
            "ref_comp": reverse_complement(ref_seq),
            "hap1": {
                "mut_seq": hap1_seq,
                "mut_comp": reverse_complement(hap1_seq),
            },
            "hap2": {
                "mut_seq": hap2_seq,
                "mut_comp": reverse_complement(hap2_seq),
            },
        }
