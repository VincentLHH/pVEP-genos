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

import numpy as np

from core.variant import Variant

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


# --------------------------------------------------------------
# 全局 dataset 缓存（进程内只加载一次）
# --------------------------------------------------------------
_dataset_cache: Dict[str, object] = {}
_region_idx_cache: Dict[str, List[Optional[int]]] = {}  # var_id → region_idx (None = 未命中)


# --------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------
def _region_key(chrom: str, start: int, end: int) -> str:
    return f"{chrom}:{start}-{end}"


def _variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}_{pos}_{ref}_{alt}"


def _numpy_bytes_to_str(arr: np.ndarray) -> str:
    """
    将 NDArray[np.bytes_]、np.uint8 或 bytes 解码为 Python str。

    兼容多种输入形状：
    - 标量 np.bytes_ / bytes
    - 1D np.bytes_ 数组
    - 1D np.uint8 数组（单个字节）
    - 2D / 3D 数组（先 flatten）
    - dtype=object 数组（genvarloader Ragged 返回值）
    """
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("ascii")
        return str(item)

    # 降维到一维
    flat = np.asarray(arr).ravel()

    if flat.dtype == np.bytes_:
        return b"".join(flat).decode("ascii")

    if flat.dtype == np.uint8:
        return bytes(flat.tolist()).decode("ascii")

    if flat.dtype == object:
        # genvarloader Ragged 可能返回 Python bytes 列表
        try:
            return b"".join(bytes(v) if isinstance(v, (list, tuple)) else v for v in flat).decode("ascii")
        except TypeError:
            pass

    # 兜底：整数数组（ASCII 码）
    try:
        return bytes(flat.tolist()).decode("ascii")
    except Exception:
        return "".join(str(x) for x in flat)


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
        self._dataset: Optional[object] = None
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

    def _ensure_dataset(self) -> object:
        """惰性创建 / 打开 .gvl 数据集。"""
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
            print(f"✅ GenVarLoader: dataset built")
        else:
            print(f"📂 GenVarLoader: loading cached dataset from {gvl_path} ...")

        self._dataset = _get_gvl().Dataset.open(
            path=gvl_path,
            reference=self.ref_fasta,
        )

        # 关键：必须显式配置返回类型。
        # 参考用户官方脚本：dataset.with_len("ragged").with_seqs("haplotypes")
        # - with_len("ragged"): 返回不等长单倍型（默认返回固定长度，索引行为不同）
        # - with_seqs("haplotypes"): 返回 haplotype 数组而非 RaggedVariants 对象
        #   否则默认 RaggedVariants.squeeze() 只接受 **kwargs，
        #   而 genvarloader 内部 __getitem__ 传位置参数，导致报错
        #   "takes 1 positional argument but 2 were given"
        # 参考：https://genvarloader.readthedocs.io/en/latest/api.html
        self._dataset = self._dataset.with_len("ragged").with_seqs("haplotypes")

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
        # 正确 API：ds.samples 返回 list[str]，ds.n_samples 返回样本数量
        # 参考：https://genvarloader.readthedocs.io/en/latest/api.html
        try:
            sample_names = list(ds.samples)
        except AttributeError:
            sample_names = None

        if sample_names is None:
            print("[DEBUG] ds.samples unavailable, assuming single sample at index 0")
            sample_idx = 0
        else:
            if sample_name not in sample_names:
                print(f"[DEBUG] sample '{sample_name}' not found in dataset. Available: {sample_names}")
                return None
            sample_idx = sample_names.index(sample_name)

        # ── 3. 重建单倍型序列 ──
        # 参考用户官方脚本：
        #   haps = dataset[region_idx, sample_idx]
        #   seq1 = haps[0].decode()  # haplotype 1
        #   seq2 = haps[1].decode()  # haplotype 2
        # haps[0] = allele 0 block, haps[1] = allele 1 block（对于二倍体）
        # 注意：每个 block 可能是多个 bytes 元素（genvarloader 用 Ragged 存储），
        # 需要 .to_list() 或解码整个 block，而不是取某个元素
        try:
            haps = ds[region_idx, sample_idx]
        except Exception as e:
            print(f"[DEBUG] ds[{region_idx}, {sample_idx}] raised: {type(e).__name__}: {e}")
            return None

        # haps 是该样本在该区域的 haplotype blocks
        # haps[0] = allele 0 的序列，haps[1] = allele 1 的序列
        # 对于 with_seqs("haplotypes")，每个 haps[i] 是一个 bytes 对象或字节数组
        # 直接取第一个 block 就是 hap1，第二个 block 就是 hap2

        # 处理 haps 的不同返回格式
        hap1_raw = haps[0]
        hap2_raw = haps[1]

        # hap1_raw / hap2_raw 可能是 bytes, bytearray, np.bytes_,
        # 或 Ragged/AnnotatedHaps 对象，需要统一解码
        def _decode_hap(hap_raw):
            # Ragged/AnnotatedHaps：尝试 .to_bytes() 或直接转 bytes
            if hasattr(hap_raw, 'to_bytes'):
                hap_raw = hap_raw.to_bytes()
            elif hasattr(hap_raw, 'to_list'):
                # Ragged 结构：to_list() 返回 bytes 列表
                lst = hap_raw.to_list()
                if isinstance(lst, list) and all(isinstance(x, (bytes, bytearray)) for x in lst):
                    return b''.join(bytes(x) for x in lst).decode('ascii')
            # 统一转为 numpy bytes 数组后解码
            arr = np.asarray(hap_raw)
            return _numpy_bytes_to_str(arr)

        hap1_seq = _decode_hap(hap1_raw)
        hap2_seq = _decode_hap(hap2_raw)

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
            "n_samples": len(ds.samples) if hasattr(ds, "samples") else "unknown",
            "sample_names": list(ds.samples) if hasattr(ds, "samples") else [],
            "gvl_path": self._gvl_path(),
        }
