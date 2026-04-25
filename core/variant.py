# core/variant.py

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


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

    @property
    def end(self) -> int:
        """
        变异占用的参考基因组终止位置（1-based，右闭）。

        - SNP: end = pos
        - 插入 (ins): alt 更长，终止仍在 pos + len(ref) - 1
        - 缺失 (del): end = pos + len(ref) - 1
        - MNV/复杂: end = pos + len(ref) - 1
        """
        return self.pos + len(self.ref) - 1

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

    def hap1_has_alt(self) -> bool:
        """单倍型1（gt[0]）是否携带 ALT 等位基因"""
        return self.gt is not None and self.gt[0] == 1

    def hap2_has_alt(self) -> bool:
        """单倍型2（gt[1]）是否携带 ALT 等位基因"""
        return self.gt is not None and self.gt[1] == 1

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
# BED 行数据类
# =========================

@dataclass
class BedRow:
    """
    表示 BED 文件中的一行（0-based 左闭右开坐标）。

    第四列 name 格式：
    - 原始行：chr_pos_ref_alt（与 Variant.id 一致，1-based）
    - 拆分后：chr_pos_ref_alt_upstream 或 chr_pos_ref_alt_downstream
    """
    chrom: str
    start: int   # 0-based, 左闭
    end: int     # 0-based, 右开
    name: str    # 第四列，变异名称（含方向后缀）

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_upstream(self) -> bool:
        return self.name.endswith("_upstream")

    @property
    def is_downstream(self) -> bool:
        return self.name.endswith("_downstream")

    @property
    def var_name(self) -> str:
        """去掉 _upstream/_downstream 后缀的变异名称"""
        if self.is_upstream:
            return self.name[: -len("_upstream")]
        if self.is_downstream:
            return self.name[: -len("_downstream")]
        return self.name

    def __repr__(self):
        return f"BedRow({self.chrom}:{self.start}-{self.end}, {self.name})"


@dataclass
class VariantBedSplit:
    """
    对应一个变异的上下游拆分结果，包含两行 BedRow。

    设计说明
    --------
    对于 1-based 变异 POS (起始), END (终止，= POS + len(ref) - 1)：

    upstream 行（正链方向，含变异上游 n bp，变异区域，变异下游 1 bp）：
        start_0based = POS - n - 1   (0-based)
        end_0based   = END + 1       (0-based, 右开)

    downstream 行（正链方向，含变异上游 1 bp，变异区域，变异下游 n bp）：
        start_0based = POS - 2       (0-based)
        end_0based   = END + n       (0-based, 右开)

    推理时：
    - upstream  正链输入 → 取末尾 w 个 token emb pooling
    - downstream 取反向互补链 → 取末尾 w 个 token emb pooling
    """
    upstream: BedRow
    downstream: BedRow
    var_name: str   # 对应的 Variant.id（不含方向后缀）


# =========================
# BED 拆分函数
# =========================

def split_variant_to_bed(
    variant: Variant,
    n: int,
    validate_n_min: Optional[int] = None,
) -> VariantBedSplit:
    """
    将单个变异按上游/下游规则拆为两行 BED。

    参数
    ----
    variant        : 待拆分的变异（需要 pos 和 ref）
    n              : 侧翼扩展长度（bp）
    validate_n_min : 若指定，n < 该值时抛出 ValueError

    坐标（0-based）
    ----
    upstream  : [POS-n-1, END+1)  即 start=POS-n-1, end=END+1（均0-based）
    downstream: [POS-2,   END+n)  即 start=POS-2,   end=END+n（均0-based）

    其中：
        POS = variant.pos  (1-based)
        END = variant.end  (1-based)
    """
    if validate_n_min is not None and n < validate_n_min:
        raise ValueError(
            f"n={n} is smaller than the minimum required n={validate_n_min}. "
            f"n must be >= max_half_window + max(len(alt), len(ref)) + 2."
        )

    pos = variant.pos   # 1-based
    end = variant.end   # 1-based

    # 转为 0-based 坐标
    up_start = pos - n - 1    # = POS - n - 1 (0-based)
    up_end   = end + 1        # = END + 1 (0-based, 右开)

    dn_start = pos - 2        # = POS - 2 (0-based)
    dn_end   = end + n        # = END + n (0-based, 右开)

    if up_start < 0:
        raise ValueError(
            f"Variant {variant.id}: upstream start ({up_start}) < 0. "
            f"Consider reducing n or filtering variants near chromosome start."
        )

    var_name = variant.id

    upstream = BedRow(
        chrom=variant.chrom,
        start=up_start,
        end=up_end,
        name=f"{var_name}_upstream",
    )
    downstream = BedRow(
        chrom=variant.chrom,
        start=dn_start,
        end=dn_end,
        name=f"{var_name}_downstream",
    )

    return VariantBedSplit(
        upstream=upstream,
        downstream=downstream,
        var_name=var_name,
    )


def parse_var_name_from_bed_name(bed_name: str) -> Tuple[str, str, str, str, str]:
    """
    从 BED 第四列名称解析变异信息。

    格式：{chrom}_{pos}_{ref}_{alt}_upstream 或 _downstream

    返回：(chrom, pos_str, ref, alt, direction)
    其中 direction 为 "upstream" 或 "downstream"（若无后缀则为空字符串）

    示例
    ----
    "chr2_202_A_AT_upstream" → ("chr2", "202", "A", "AT", "upstream")
    "chr2_202_A_AT"          → ("chr2", "202", "A", "AT", "")
    """
    direction = ""
    name = bed_name
    for suffix in ("_upstream", "_downstream"):
        if bed_name.endswith(suffix):
            direction = suffix[1:]  # 去掉前缀 _
            name = bed_name[: -len(suffix)]
            break

    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot parse variant name from BED name: {bed_name!r}. "
            f"Expected format: {{chrom}}_{{pos}}_{{ref}}_{{alt}}[_upstream|_downstream]"
        )

    # chrom 可能含 _ (如 chr_un_xxx)，但通常是 chrN，保守处理：
    # 从右侧取 ref, alt，再取 pos（数字），其余拼为 chrom
    alt = parts[-1]
    ref = parts[-2]
    pos_str = parts[-3]
    chrom = "_".join(parts[:-3])

    return chrom, pos_str, ref, alt, direction


def load_bed(bed_path: str) -> List[BedRow]:
    """
    读取 BED 文件，返回 BedRow 列表。
    支持 3 列（无名称）和 4+ 列（有名称）格式。
    """
    rows: List[BedRow] = []
    with open(bed_path) as f:
        for line_no, line in enumerate(f, 1):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) >= 4 else f"{chrom}_{start}_{end}"
            rows.append(BedRow(chrom=chrom, start=start, end=end, name=name))
    return rows


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
