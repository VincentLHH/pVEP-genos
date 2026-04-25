"""
core/genvarloader_builder.py
=============================
GenVarLoader 驱动的序列构建器（新版）。

设计原则
--------
1. 完全依赖 GenVarLoader 提供 Mut_hap 序列（不使用 pysam 手写 indel 对齐）；
2. 从 BedRow 的第四列名称（chr_pos_ref_alt_upstream/downstream）反推变异信息；
3. 基于反推的变异，对 genvarloader 返回的真实单倍型进行 WT_hap 恢复；
4. WT_ref / Mut_ref 由 pysam 直接从参考基因组构建，不依赖 genvarloader；
5. 返回与 SequenceBuilder 完全对齐的 SixSeqResult 结构。

WT_hap 恢复逻辑（方案B）
------------------------
genvarloader 返回的序列即是真实单倍型（Mut_hap 如果携带了目标变异，
否则就是背景序列等价于 WT_hap）。

对于携带了目标变异的单倍型，需要从序列末尾定位目标变异区域并替换回 ref：
- upstream  行：目标变异区域在序列末尾附近（最后 var_len + 1 bp），
  从序列末尾向前数 1 位（因为末尾设计了 1bp 下游 padding）
- downstream 行：目标变异区域在序列开头附近（前 var_len + 1 bp），
  从序列开头向后数 1 位（因为开头设计了 1bp 上游 padding）

具体坐标：
    upstream 行（长度 n + var_len + 1）：
        变异区域在序列中的位置：[-（var_len+1）:-1]（不含最后1bp）
        即 offset = len(seq) - var_len - 1
    downstream 行（长度 var_len + 1 + n）：
        变异区域在序列中的位置：[1 : 1+var_len]
        即 offset = 1
"""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

from core.variant import (
    BedRow,
    Variant,
    VariantBedSplit,
    parse_var_name_from_bed_name,
)
from core.sequence_builder import (
    HapSeqPair,
    SixSeqResult,
    reverse_complement,
)

# 延迟导入 genvarloader（可能未安装），首次使用时才检查
_gvl_module: Optional[object] = None


def _get_gvl():
    global _gvl_module
    if _gvl_module is None:
        try:
            import genvarloader as _g
            _gvl_module = _g
        except ImportError:
            raise ImportError(
                "genvarloader is not installed. "
                "Please install: pip install genvarloader"
            )
    return _gvl_module


# ===========================================================================
# 工具函数
# ===========================================================================

def _numpy_bytes_to_str(arr: np.ndarray) -> str:
    """
    将 NDArray[np.bytes_]、np.uint8 或 bytes 解码为 Python str。
    """
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("ascii")
        return str(item)

    flat = np.asarray(arr).ravel()

    if flat.dtype == np.bytes_:
        return b"".join(flat).decode("ascii")

    if flat.dtype == np.uint8:
        return bytes(flat.tolist()).decode("ascii")

    if flat.dtype == object:
        try:
            return b"".join(
                bytes(v) if isinstance(v, (list, tuple)) else v for v in flat
            ).decode("ascii")
        except TypeError:
            pass

    try:
        return bytes(flat.tolist()).decode("ascii")
    except Exception:
        return "".join(str(x) for x in flat)


def _decode_hap(hap_raw) -> str:
    """统一解码 genvarloader 返回的单倍型数据为 Python str。"""
    if hasattr(hap_raw, "to_bytes"):
        hap_raw = hap_raw.to_bytes()
    elif hasattr(hap_raw, "to_list"):
        lst = hap_raw.to_list()
        if isinstance(lst, list) and all(
            isinstance(x, (bytes, bytearray)) for x in lst
        ):
            return b"".join(bytes(x) for x in lst).decode("ascii")
    arr = np.asarray(hap_raw)
    return _numpy_bytes_to_str(arr)


# ===========================================================================
# 主类
# ===========================================================================

class GenVarLoaderSequenceBuilder:
    """
    基于 GenVarLoader 的序列构建器（新版，适配 6 种序列框架）。

    参数
    ----
    vcf_path       : VCF 文件路径（.vcf.gz 或 .vcf）
    bed_path       : BED 文件路径（已拆分为 upstream/downstream 行）
    ref_fasta      : 参考基因组 FASTA 路径
    gvl_cache_dir  : .gvl 数据集缓存目录（默认 /tmp/gvl_cache）
    strandaware    : 是否对负链区域做 reverse complement（默认 False，
                     我们手动控制方向，不让 GVL 自动处理）
    max_mem        : gvl.write 内存上限（默认 '4g'）
    overwrite      : 是否强制覆盖已有 .gvl 文件（默认 False）
    window_size    : 模型推理上下文窗口大小（仅用于日志提示）
    n              : BED 拆分侧翼长度（与 BedRow 中的 n 保持一致，用于反算变异偏移）
    """

    def __init__(
        self,
        vcf_path: str,
        bed_path: str,
        ref_fasta: str,
        gvl_cache_dir: str = "/tmp/gvl_cache",
        strandaware: bool = False,
        max_mem: str = "4g",
        overwrite: bool = False,
        window_size: int = 128,
        n: int = 200,
    ):
        self.vcf_path = vcf_path
        self.bed_path = bed_path
        self.ref_fasta = ref_fasta
        self.gvl_cache_dir = gvl_cache_dir
        self.strandaware = strandaware
        self.max_mem = max_mem
        self.overwrite = overwrite
        self.window_size = window_size
        self.n = n
        self._half_window = window_size // 2

        self._fasta: Optional[pysam.FastaFile] = None
        self._dataset: Optional[object] = None
        self._bed_rows: List[BedRow] = []
        # name → region_idx 的映射（name 是 BedRow 的 name 字段）
        self._name_to_idx: Dict[str, int] = {}
        # 当前处理的样本名（由 process_all 注入）
        self.current_sample: Optional[str] = None

    # =========================================================
    # 惰性初始化
    # =========================================================

    def _gvl_path(self) -> str:
        """根据 vcf + bed 内容生成确定性的 .gvl 文件路径。"""
        os.makedirs(self.gvl_cache_dir, exist_ok=True)
        key = hashlib.md5(
            f"{self.vcf_path}{self.bed_path}".encode()
        ).hexdigest()[:12]
        return os.path.join(self.gvl_cache_dir, f"{key}.gvl")

    def _ensure_fasta(self) -> pysam.FastaFile:
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.ref_fasta)
        return self._fasta

    def _ensure_dataset(self) -> object:
        """惰性创建 / 打开 .gvl 数据集，并建立 name → region_idx 映射。"""
        if self._dataset is not None:
            return self._dataset

        gvl_path = self._gvl_path()

        if not os.path.exists(gvl_path) or self.overwrite:
            print(f"🔨 GenVarLoader: building dataset at {gvl_path} ...")
            _get_gvl().write(
                path=gvl_path,
                bed=self.bed_path,
                variants=self.vcf_path,
                max_mem=self.max_mem,
                overwrite=self.overwrite,
            )
            print("✅ GenVarLoader: dataset built")
        else:
            print(f"📂 GenVarLoader: loading cached dataset from {gvl_path} ...")

        ds = _get_gvl().Dataset.open(
            path=gvl_path,
            reference=self.ref_fasta,
        )

        # with_len("ragged")：返回不等长单倍型（per-region 实际长度）
        # with_seqs("haplotypes")：返回 haplotype bytes 数组
        ds = ds.with_len("ragged").with_seqs("haplotypes")

        if not self.strandaware:
            # 我们手动控制链方向，禁止 GVL 自动 reverse complement
            ds = ds.with_settings(rc_neg=False)

        self._dataset = ds

        # 从 dataset.regions 获取 name 列表，建立 name → index 映射
        # 参考用户官方脚本：
        #   vars = dataset.regions["name"].to_list()
        try:
            names = self._dataset.regions["name"].to_list()
            self._name_to_idx = {name: idx for idx, name in enumerate(names)}
            print(f"[GVL] Loaded {len(names)} regions from dataset.")
        except Exception as e:
            warnings.warn(
                f"[GVL] Failed to read dataset.regions['name']: {e}. "
                "Will fall back to positional index lookup."
            )
            # 回退：直接读取 bed 文件建立映射
            self._bed_rows = self._load_bed_rows()
            self._name_to_idx = {
                row.name: idx for idx, row in enumerate(self._bed_rows)
            }

        return self._dataset

    def _load_bed_rows(self) -> List[BedRow]:
        """解析 BED 文件，返回 BedRow 列表（含第四列 name）。"""
        from core.variant import load_bed
        return load_bed(self.bed_path)

    # =========================================================
    # 核心构建接口（新版）
    # =========================================================

    def build_six_seqs(
        self,
        bed_split: VariantBedSplit,
        center_variant: Variant,
        sample_name: Optional[str] = None,
    ) -> Optional[Dict[str, SixSeqResult]]:
        """
        对 VariantBedSplit 中的上游和下游 BedRow 各构建 SixSeqResult。

        返回
        ----
        {"upstream": SixSeqResult, "downstream": SixSeqResult}
        若任意一行构建失败则返回 None。
        """
        sname = sample_name or self.current_sample
        if sname is None:
            raise ValueError(
                "sample_name must be provided either as parameter or "
                "via builder.current_sample"
            )

        up_result = self.build_from_bed(bed_split.upstream, center_variant, sname)
        if up_result is None:
            return None

        dn_result = self.build_from_bed(bed_split.downstream, center_variant, sname)
        if dn_result is None:
            return None

        return {"upstream": up_result, "downstream": dn_result}

    def build_from_bed(
        self,
        bed_row: BedRow,
        center_variant: Variant,
        sample_name: str,
    ) -> Optional[SixSeqResult]:
        """
        基于一行 BedRow 构建 SixSeqResult。

        步骤
        ----
        1. 从 FASTA 获取 WT_ref
        2. 构建 Mut_ref（在 WT_ref 中强制注入目标变异）
        3. 通过 genvarloader 获取该样本在该行的 hap1/hap2 序列（= Mut_hap）
        4. 根据基因型决定是否需要恢复 WT_hap：
           - 携带 alt → 从 Mut_hap 中移除目标变异得到 WT_hap
           - 不携带 alt → WT_hap = Mut_hap（直接引用，标记等价）
        """
        ds = self._ensure_dataset()

        # 1. 从 FASTA 获取 WT_ref
        wt_ref = self._fetch_bed_row(bed_row)
        if wt_ref is None:
            return None

        # 2. 构建 Mut_ref
        mut_ref = self._inject_variant_into_ref(wt_ref, bed_row, center_variant)
        if mut_ref is None:
            return None

        # 3. 定位 region_idx
        region_idx = self._name_to_idx.get(bed_row.name)
        if region_idx is None:
            warnings.warn(
                f"[GVL] BedRow name '{bed_row.name}' not found in dataset. "
                f"Available names (first 5): {list(self._name_to_idx.keys())[:5]}"
            )
            return None

        # 4. 定位 sample_idx
        try:
            sample_names = list(ds.samples)
        except AttributeError:
            sample_names = None

        if sample_names is None:
            sample_idx = 0
        else:
            if sample_name not in sample_names:
                warnings.warn(
                    f"[GVL] Sample '{sample_name}' not in dataset. "
                    f"Available: {sample_names[:5]}"
                )
                return None
            sample_idx = sample_names.index(sample_name)

        # 5. 从 genvarloader 获取单倍型序列
        try:
            haps = ds[region_idx, sample_idx]
        except Exception as e:
            warnings.warn(
                f"[GVL] ds[{region_idx}, {sample_idx}] failed: {type(e).__name__}: {e}"
            )
            return None

        hap1_seq = _decode_hap(haps[0])
        hap2_seq = _decode_hap(haps[1])

        if not hap1_seq or not hap2_seq:
            return None

        # 6. 构建 hap1 和 hap2 的 Mut/WT 序列对
        direction = "upstream" if bed_row.is_upstream else "downstream"
        is_upstream = bed_row.is_upstream

        hap1_pair = self._build_hap_pair_from_gvl(
            hap_seq=hap1_seq,
            center_variant=center_variant,
            bed_row=bed_row,
            is_upstream=is_upstream,
            hap_index=0,
        )
        hap2_pair = self._build_hap_pair_from_gvl(
            hap_seq=hap2_seq,
            center_variant=center_variant,
            bed_row=bed_row,
            is_upstream=is_upstream,
            hap_index=1,
        )

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

    # =========================================================
    # 内部实现：WT_hap 恢复（方案B：在 GVL 返回的序列上定位并替换）
    # =========================================================

    def _build_hap_pair_from_gvl(
        self,
        hap_seq: str,
        center_variant: Variant,
        bed_row: BedRow,
        is_upstream: bool,
        hap_index: int,
    ) -> HapSeqPair:
        """
        从 genvarloader 返回的真实单倍型序列构建 HapSeqPair。

        逻辑
        ----
        - genvarloader 返回的就是真实单倍型，如果该 hap 携带目标变异，
          则 hap_seq 中目标位置是 alt；否则是 ref 等位基因（或背景序列）。
        - 我们无法从外部直接知道序列内容，但可以根据基因型判断：
            hap_has_alt = center_variant.gt[hap_index] == 1
          若携带，则需要从 hap_seq 中恢复 WT_hap（移除目标变异，替换回 ref）。

        定位目标变异在 hap_seq 中的偏移
        --------------------------------
        由于 genvarloader 已经处理好了全部背景变异的 indel 对齐，
        我们无法直接用参考基因组坐标来定位。但 BED 行的设计已保证：

        upstream 行：变异区域紧邻序列末尾前 1 bp
            目标变异的 alt 在 hap_seq 中的偏移 = len(hap_seq) - var_len_in_hap - 1
            其中 var_len_in_hap = len(alt) if hap_has_alt else len(ref)

        downstream 行：变异区域紧邻序列开头后 1 bp
            目标变异的 alt/ref 在 hap_seq 中的偏移 = 1

        注意：这里使用 alt 长度（而非 ref 长度），因为 hap_seq 已经是
        应用了变异之后的序列。
        """
        hap_has_alt = (
            center_variant.gt is not None
            and center_variant.gt[hap_index] == 1
        )

        if not hap_has_alt:
            # 不携带目标变异，Mut == WT
            return HapSeqPair(mut=hap_seq, wt=hap_seq, wt_is_alias_of_mut=True)

        # 携带目标变异，需要恢复 WT_hap
        wt_hap = self._restore_wt_from_mut_hap(
            mut_hap=hap_seq,
            center_variant=center_variant,
            is_upstream=is_upstream,
        )

        if wt_hap is None:
            warnings.warn(
                f"[GVL] Failed to restore WT_hap for variant {center_variant.id}, "
                f"hap_index={hap_index}. Falling back to Mut=WT."
            )
            return HapSeqPair(mut=hap_seq, wt=hap_seq, wt_is_alias_of_mut=True)

        return HapSeqPair(mut=hap_seq, wt=wt_hap, wt_is_alias_of_mut=False)

    def _restore_wt_from_mut_hap(
        self,
        mut_hap: str,
        center_variant: Variant,
        is_upstream: bool,
    ) -> Optional[str]:
        """
        从 Mut_hap（携带目标变异）恢复 WT_hap（移除目标变异，替换为 ref）。

        定位逻辑
        --------
        upstream 行（序列正向，变异在末尾前 1bp）：
            alt 在序列中的位置：len(mut_hap) - len(alt) - 1
            替换为 ref 即可

        downstream 行（序列正向，变异在开头后 1bp）：
            alt 在序列中的位置：offset = 1
            替换为 ref 即可

        Returns
        -------
        恢复后的 WT_hap 字符串，或 None（若定位越界）
        """
        ref = center_variant.ref
        alt = center_variant.alt
        alt_len = len(alt)
        ref_len = len(ref)

        if is_upstream:
            # upstream：alt 在序列末尾前 1bp
            # 末尾设计：[... alt_seq ... 下游1bp] → alt 从 len(seq)-alt_len-1 开始
            offset = len(mut_hap) - alt_len - 1
        else:
            # downstream：alt 在序列开头后 1bp
            # 开头设计：[上游1bp, alt_seq, ...] → alt 从 offset=1 开始
            offset = 1

        if offset < 0 or offset + alt_len > len(mut_hap):
            return None

        current = mut_hap[offset: offset + alt_len]

        if current.upper() != alt.upper():
            # 序列不匹配，可能是 indel 引入了额外偏移，记录警告但仍尝试恢复
            warnings.warn(
                f"[GVL] WT_hap restore: expected alt={alt!r} at offset {offset}, "
                f"found {current!r}. This may indicate background indel shift. "
                f"Proceeding with forced replacement."
            )

        wt_hap = mut_hap[:offset] + ref + mut_hap[offset + alt_len:]
        return wt_hap

    # =========================================================
    # 参考基因组操作（WT_ref / Mut_ref）
    # =========================================================

    def _fetch_bed_row(self, bed_row: BedRow) -> Optional[str]:
        """
        从 FASTA 按 BedRow 的 0-based 坐标提取序列（WT_ref）。
        pysam.fetch 接受 0-based 左闭右开坐标。
        """
        try:
            fasta = self._ensure_fasta()
            seq = fasta.fetch(bed_row.chrom, bed_row.start, bed_row.end).upper()
            return seq
        except Exception as e:
            warnings.warn(f"[GVL] FASTA fetch failed for {bed_row}: {e}")
            return None

    def _inject_variant_into_ref(
        self,
        ref_seq: str,
        bed_row: BedRow,
        variant: Variant,
    ) -> Optional[str]:
        """
        在 ref_seq 中把目标变异的 ref 等位基因替换为 alt（强制，不考虑基因型）。
        坐标换算：offset = variant.pos - 1 - bed_row.start
        """
        offset = variant.pos - 1 - bed_row.start

        if offset < 0 or offset + len(variant.ref) > len(ref_seq):
            return None

        current = ref_seq[offset: offset + len(variant.ref)]
        if current.upper() != variant.ref.upper():
            warnings.warn(
                f"[GVL] Mut_ref injection: ref mismatch for {variant.id} at offset {offset}. "
                f"Expected {variant.ref!r}, found {current!r}. Forcing injection."
            )

        return ref_seq[:offset] + variant.alt + ref_seq[offset + len(variant.ref):]

    # =========================================================
    # 兼容旧接口（已废弃）
    # =========================================================

    def build(
        self,
        center_variant: Variant,
        all_variants: List[Variant],
        sample_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        [已废弃] 旧版 build() 接口，仅保留兼容旧测试。
        新代码请使用 build_six_seqs() 或 build_from_bed()。
        """
        warnings.warn(
            "GenVarLoaderSequenceBuilder.build() is deprecated. "
            "Use build_six_seqs() or build_from_bed() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        sname = sample_name or self.current_sample
        if sname is None:
            return None

        ds = self._ensure_dataset()
        chrom = center_variant.chrom
        pos = center_variant.pos

        region_idx = None
        for name, idx in self._name_to_idx.items():
            # 通过 region 坐标范围确认
            if chrom in name and str(pos) in name:
                region_idx = idx
                break

        # Fallback：如果 name 匹配失败，尝试从 dataset 的第一个 region 获取
        if region_idx is None:
            try:
                names = self._dataset.regions["name"].to_list()
                if names:
                    # 旧版 BED（无第4列）时，regions["name"] 可能就是 BED 行文本
                    # 直接取第一个 region 作为 fallback
                    region_idx = 0
            except Exception:
                pass

        if region_idx is None:
            return None

        try:
            sample_names = list(ds.samples)
        except AttributeError:
            sample_names = None

        if sample_names is None:
            sample_idx = 0
        else:
            if sname not in sample_names:
                return None
            sample_idx = sample_names.index(sname)

        try:
            haps = ds[region_idx, sample_idx]
        except Exception:
            return None

        hap1_seq = _decode_hap(haps[0])
        hap2_seq = _decode_hap(haps[1])

        if not hap1_seq or not hap2_seq:
            return None

        fasta = self._ensure_fasta()
        left = pos - self._half_window
        right = pos + len(center_variant.ref) - 1 + self._half_window
        if left < 1:
            return None
        try:
            ref_seq = fasta.fetch(center_variant.chrom, left - 1, right).upper()
        except Exception:
            return None

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

    # =========================================================
    # 诊断接口
    # =========================================================

    def get_dataset_info(self) -> Dict:
        """返回数据集统计信息（用于 debug）。"""
        ds = self._ensure_dataset()
        return {
            "n_regions": len(self._name_to_idx),
            "n_samples": len(ds.samples) if hasattr(ds, "samples") else "unknown",
            "sample_names": list(ds.samples) if hasattr(ds, "samples") else [],
            "gvl_path": self._gvl_path(),
        }
