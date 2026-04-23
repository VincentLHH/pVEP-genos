# core/variant.py

from typing import Tuple, Optional


class Variant:
    """
    表示一个变异（VCF record级别）
    """

    def __init__(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        gt: Optional[Tuple[int, int]] = None
    ):
        self.chrom = chrom
        self.pos = pos
        self.ref = ref
        self.alt = alt
        self.gt = gt  # (0,1), (1,0), (1,1), (0,0), or None

    @property
    def id(self) -> str:
        return f"{self.chrom}_{self.pos}_{self.ref}_{self.alt}"

    def is_non_reference(self) -> bool:
        """
        是否为非 0|0 变异
        """
        if self.gt is None or None in self.gt:
            return False
        return any(allele == 1 for allele in self.gt)

    def is_homozygous_alt(self) -> bool:
        return self.gt == (1, 1)

    def is_heterozygous(self) -> bool:
        return self.gt in [(1, 0), (0, 1)]

    def get_haplotypes(self):
        """
        返回两个 haplotype 对象
        """
        if self.gt is None:
            return None, None
        return Haplotype(self.gt[0]), Haplotype(self.gt[1])

    def __repr__(self):
        return f"Variant({self.id}, GT={self.gt})"


class Haplotype:
    """
    表示单倍型（0 或 1）
    """

    def __init__(self, allele: int):
        self.allele = allele  # 0=ref, 1=alt

    def is_ref(self) -> bool:
        return self.allele == 0

    def is_alt(self) -> bool:
        return self.allele == 1

    def __repr__(self):
        return f"Haplotype({self.allele})"


# =========================
# 工具函数（VCF解析相关）
# =========================

def parse_gt(gt_field) -> Optional[Tuple[int, int]]:
    """
    从 pysam record.samples[sample]['GT'] 解析 genotype

    支持：
    (1,0)
    (0,1)
    (1,1)
    (0,0)
    (None, 1)
    """

    if gt_field is None:
        return None

    if not isinstance(gt_field, tuple):
        return None

    if len(gt_field) != 2:
        return None

    a, b = gt_field

    if a is None or b is None:
        return None

    # 只保留二等位（你当前设计）
    if a > 1 or b > 1:
        return None

    return int(a), int(b)
