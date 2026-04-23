from typing import List, Dict, Optional
import pysam

from core.variant import Variant


class SequenceBuilder:
    def __init__(self, fasta_path: str, window_size: int = 128):
        self.fasta = pysam.FastaFile(fasta_path)
        self.window_size = window_size
        self.half_window = window_size // 2

    def reverse_complement(self, seq: str) -> str:
        complement = str.maketrans("ATCGatcg", "TAGCtagc")
        return seq[::-1].translate(complement)

    def build(
        self,
        center_variant: Variant,
        variants_in_region: List[Variant],  # ⚠️ 关键：整个窗口内的所有variant
    ) -> Optional[Dict]:

        # === 1. 获取参考窗口 ===
        ref_seq, left = self._get_ref_window(center_variant)
        if ref_seq is None:
            return None

        # === 2. 过滤窗口内变异 ===
        region_vars = self._filter_variants(
            variants_in_region,
            left,
            left + len(ref_seq) - 1
        )

        # === 3. 构建 hap1 / hap2 ===
        hap1_seq = self._apply_variants(ref_seq, left, region_vars, hap_index=0)
        hap2_seq = self._apply_variants(ref_seq, left, region_vars, hap_index=1)

        # === 4. 构建互补链 ===
        return {
            "ref_seq": ref_seq,
            "ref_comp": self.reverse_complement(ref_seq),

            "hap1": {
                "mut_seq": hap1_seq,
                "mut_comp": self.reverse_complement(hap1_seq)
            },
            "hap2": {
                "mut_seq": hap2_seq,
                "mut_comp": self.reverse_complement(hap2_seq)
            }
        }

    def _get_ref_window(self, variant: Variant):
        pos = variant.pos
        left = pos - self.half_window
        right = pos + len(variant.ref) - 1 + self.half_window

        if left < 1:
            return None, None

        try:
            seq = self.fasta.fetch(variant.chrom, left - 1, right).upper()
        except Exception:
            return None, None

        return seq, left

    def _filter_variants(self, variants, left, right):
        return [
            v for v in variants
            if left <= v.pos <= right and v.gt is not None
        ]

    def _apply_variants(self, ref_seq, left, variants, hap_index):

        seq = ref_seq
        shift = 0

        variants_sorted = sorted(variants, key=lambda v: v.pos)

        for v in variants_sorted:

            allele = v.gt[hap_index]

            if allele == 0:
                continue

            ref = v.ref
            alt = v.alt

            pos_in_seq = v.pos - left + shift

            current = seq[pos_in_seq:pos_in_seq + len(ref)]

            # 严格校验 REF 匹配
            if current != ref:
                continue

            # 替换
            seq = (
                    seq[:pos_in_seq]
                    + alt
                    + seq[pos_in_seq + len(ref):]
            )

            print(f"   ✔ applied: {ref} -> {alt}")

            # 更新 shift
            shift += len(alt) - len(ref)

        return seq
