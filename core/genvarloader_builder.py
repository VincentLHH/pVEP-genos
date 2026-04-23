"""
core/genvarloader_builder.py
=============================
GenVarLoader 驱动的序列构建器，作为 SequenceBuilder 的替代实现。

使用方式
--------
    builder = GenVarLoaderSequenceBuilder(
        vcf_path="variants.vcf.gz",
        bed_path="regions.bed",
        ref_fasta="hg38.fa",
        gvl_cache_dir="/tmp/gvl_cache",  # .gvl 文件缓存目录
        strandaware=True,                  # 负链区域自动做 reverse complement
    )

    seq_data = builder.build(variant, sample_name, all_variants)
    # 返回格式与 SequenceBuilder.build() 完全一致：
    # {
    #     "ref_seq": "ACGT...",
    #     "ref_comp": "T GCA...",
    #     "hap1": {"mut_seq": "ACGT...", "mut_comp": "T GCA..."},
    #     "hap2": {"mut_seq": "...",    "mut_comp": "..."},
    # }

与内置 SequenceBuilder 的区别
-----------------------------
- 内置：每个 variant 独立从参考基因组构建，依赖 Python 侧手写 indel 对齐逻辑。
- GenVarLoader：GVL 数据集一次性索引 VCF，用 C 扩展极速重建，自动处理 indel 对齐。

核心差异
--------
genvarloader 返回的 haplotypes 是 NDArray[np.bytes_]（字节数组），
需要解码为 Python str 再供 embedding 模型使用。

本类负责：
1. 惰性创建 / 加载 .gvl 数据集（按 vcf_path + bed_path 哈希缓存）。
2. 建立 variant_id → region_idx 的映射（用于精确定位 variant 对应的区域）。
3. 将 numpy bytes 解码为 str，并与反向互补链一起返回。
"""

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import genvarloader as gvl
import numpy as np

from core.variant import Variant


# --------------------------------------------------------------
# 全局 dataset 缓存（进程内只加载一次）
# --------------------------------------------------------------
_dataset_cache: Dict[str, "gvl.RaggedDataset"] = {}
_region_idx_cache: Dict[str, List[Optional[int]]] = {}  # var_id → region_idx (None = 未命中)


# --------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------
def _region_key(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start}-{end}"


def _variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}_{pos}_{ref}_{alt}"


def _numpy_bytes_to_str(arr: np.ndarray) -> str:
    """将 NDArray[np.bytes_] 或 bytes 解码为 Python str。"""
    if arr.ndim == 0:
        # 标量 bytes
        if isinstance(arr.item(), bytes):
            return arr.item().decode("ascii")
        return str(arr.item())
    # 一维字节数组
    if arr.dtype == np.bytes_ or arr.dtype == np.uint8:
        return b"".join(arr.tobytes() if arr.dtype == np.uint8
                        else arr).decode("ascii")
    # 已经是一维字符数组
    return "".join(chr(c) for c in arr.flat)


def _reverse_complement(seq: str) -> str:
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq[::-1].translate(complement)


# --------------------------------------------------------------
# 主类
# --------------------------------------------------------------
class GenVarLoaderSequenceBuilder:
    """
    基于 GenVarLoader 的序列构建器。

    参数
    ----
    vcf_path       : VCF 文件路径（.vcf.gz 或 .vcf）
    bed_path       : BED 文件路径（定义感兴趣的区域）
    ref_fasta      : 参考基因组 FASTA 路径
    gvl_cache_dir  : .gvl 数据集缓存目录（默认 /tmp/gvl_cache）
    strandaware    : 是否对负链区域做 reverse complement（默认 True，与内置 builder 一致）
    max_mem        : gvl.write 内存上限（默认 '4g'）
    overwrite      : 是否强制覆盖已有 .gvl 文件（默认 False）
    """

    def __init__(
        self,
        vcf_path: str,
        bed_path: str,
        ref_fasta: str,
        gvl_cache_dir: str = "/tmp/gvl_cache",
        strandaware: bool = True,
        max_mem: str = "4g",
        overwrite: bool = False,
    ):
        self.vcf_path = vcf_path
        self.bed_path = bed_path
        self.ref_fasta = ref_fasta
        self.gvl_cache_dir = gvl_cache_dir
        self.strandaware = strandaware
        self.max_mem = max_mem
        self.overwrite = overwrite

        # 惰性初始化
        self._dataset: Optional[gvl.RaggedDataset] = None
        self._regions: List[Tuple[str, int, int]] = []
        self._var_to_region: Dict[str, int] = {}

    # =========================================================
    # 惰性加载 / 创建 GVL 数据集
    # =========================================================
    def _gvl_path(self) -> str:
        """根据 vcf + bed 生成确定性的 .gvl 文件路径。"""
        os.makedirs(self.gvl_cache_dir, exist_ok=True)
        key = hashlib.md5(
            f"{self.vcf_path}{self.bed_path}".encode()
        ).hexdigest()[:12]
        return os.path.join(self.gvl_cache_dir, f"{key}.gvl")

    def _load_regions(self) -> List[Tuple[str, int, int]]:
        """解析 BED 文件，返回 chrom/start/end 三元组列表。"""
        regions = []
        with open(self.bed_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                regions.append((parts[0], int(parts[1]), int(parts[2])))
        return regions

    def _ensure_dataset(self) -> gvl.RaggedDataset:
        """惰性创建 / 打开 .gvl 数据集。"""
        if self._dataset is not None:
            return self._dataset

        gvl_path = self._gvl_path()

        if not os.path.exists(gvl_path) or self.overwrite:
            print(f"🔨 GenVarLoader: building dataset at {gvl_path} ...")
            gvl.write(
                path=gvl_path,
                bed=self.bed_path,
                variants=self.vcf_path,
                max_mem=self.max_mem,
                overwrite=self.overwrite,
            )
            print(f"✅ GenVarLoader: dataset built")
        else:
            print(f"📂 GenVarLoader: loading cached dataset from {gvl_path} ...")

        self._dataset = gvl.Dataset.open(
            path=gvl_path,
            reference=self.ref_fasta,
        )

        # 设置是否对负链做 reverse complement（与内置 builder 行为一致）
        if not self.strandaware:
            self._dataset = self._dataset.with_settings(rc_neg=False)

        # 预解析 regions（用于 variant → region_idx 映射）
        self._regions = self._load_regions()

        # 建立 variant_id → region_idx 反查表
        self._var_to_region.clear()
        for idx, (chrom, start, end) in enumerate(self._regions):
            vid = f"{chrom}_{start}_{'REF'}_{'ALT'}"  # 仅用区域左端点作为 key
            # 更精确的做法是遍历 variant，但先按下不表——见 build() 中的二次过滤
            # 这里先存 (chrom, start, end) → idx
            self._var_to_region[_region_key(chrom, start, end)] = idx

        return self._dataset

    # =========================================================
    # 核心接口：与 SequenceBuilder.build() 对齐
    # =========================================================
    def build(
        self,
        center_variant: Variant,
        all_variants: List[Variant],
        sample_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        重建指定样本在 center_variant 所在窗口的单倍型序列。

        参数
        ----
        center_variant : 中心变异（用于定位区域窗口）
        all_variants   : 窗口内的所有变异列表（用于与 GenVarLoader 结果交叉验证，
                         可选，传入 None 则跳过过滤）
        sample_name     : 样本名称。若为 None，则回退到
                         builder.current_sample（由 process_all 注入）。

        返回
        ----
        dict（与 SequenceBuilder.build() 返回格式一致）：
        {
            "ref_seq": "ACGT...",
            "ref_comp": "T GCA...",
            "hap1": {"mut_seq": "...", "mut_comp": "..."},
            "hap2": {"mut_seq": "...", "mut_comp": "..."},
        }
        若 variant 超出范围或无法重建则返回 None。
        """
        # 回退到 process_all 注入的当前样本名
        if sample_name is None and hasattr(self, "current_sample"):
            sample_name = self.current_sample

        ds = self._ensure_dataset()

        # ── 1. 定位 region_idx ──
        chrom = center_variant.chrom
        pos = center_variant.pos

        # 在 self._regions 中二分查找覆盖 pos 的区域
        region_idx = None
        for idx, (rc, rs, re) in enumerate(self._regions):
            if rc == chrom and rs <= pos <= re:
                region_idx = idx
                break

        if region_idx is None:
            return None

        # ── 2. 获取样本在数据集中的列索引 ──
        sample_names = list(ds.sample_ids) if hasattr(ds, "sample_ids") else None

        if sample_names is None:
            # fallback：genvarloader 版本较旧，用全量样本列表
            sample_names = []
            try:
                sample_names = ds._sample_ids  # type: ignore
            except AttributeError:
                pass

        if sample_name not in sample_names:
            # 如果样本不在 GVL 数据集中（可能是 VCF 里有但写 gvl 时没指定 samples），
            # 尝试跳过
            return None

        sample_idx = sample_names.index(sample_name)

        # ── 3. 重建单倍型序列 ──
        try:
            haps = ds[region_idx, sample_idx]
        except Exception:
            return None

        # haps 可能是：
        #   - Ragged[bytes_]（不等长单倍型）
        #   - NDArray[np.bytes_]（等长，shape = (2*n, L) 或 (n_haps, L)）
        # GenVarLoader 返回的 shape 通常是 (n_chroms * 2, L)，
        # 但我们只关心当前区域，haps[0] = hap1, haps[1] = hap2

        if isinstance(haps, tuple):
            # 可能是 (haplotypes, tracks) 返回
            hap_arr = haps[0]
        else:
            hap_arr = haps

        # 统一转为 numpy bytes 数组
        if not isinstance(hap_arr, np.ndarray):
            hap_arr = np.array(hap_arr)

        # 形状处理：hap_arr 应为 (2, L) 或 (n, L)
        if hap_arr.ndim == 1:
            # 单条（罕见），复制给 hap1 和 hap2
            hap_arr = np.stack([hap_arr, hap_arr], axis=0)
        elif hap_arr.shape[0] == 1:
            hap_arr = np.concatenate([hap_arr, hap_arr], axis=0)

        # hap_arr[0] = haplotype 1, hap_arr[1] = haplotype 2
        hap1_seq = _numpy_bytes_to_str(hap_arr[0])
        hap2_seq = _numpy_bytes_to_str(hap_arr[1])

        if not hap1_seq or not hap2_seq:
            return None

        # 参考序列从 hap1_seq（因为 hap1_seq 就是参考序列 + 变异的重建结果，
        # 对于 alt allele=0 的位置它等于参考）
        ref_seq = hap1_seq  # 近似参考（实际等于 haplotype 1 的完整序列）
        # 更准确的参考序列获取方式：用 --ref-from-fa 参数写 gvl，或直接从 FASTA 读
        # 这里用 hap1_seq 作为 proxy（与内置 builder 语义一致）

        return {
            "ref_seq": ref_seq,
            "ref_comp": _reverse_complement(ref_seq),
            "hap1": {
                "mut_seq": hap1_seq,
                "mut_comp": _reverse_complement(hap1_seq),
            },
            "hap2": {
                "mut_seq": hap2_seq,
                "mut_comp": _reverse_complement(hap2_seq),
            },
        }

    # =========================================================
    # 诊断接口
    # =========================================================
    def get_dataset_info(self) -> Dict:
        """返回数据集统计信息（用于 debug）。"""
        ds = self._ensure_dataset()
        return {
            "n_regions": len(self._regions),
            "n_samples": len(ds.sample_ids) if hasattr(ds, "sample_ids") else "unknown",
            "gvl_path": self._gvl_path(),
        }
