"""
core/embedding_extractor.py
============================
个性化 VEP 的核心 embedding 提取器。

设计
----
对每一个变异（及对应的 VariantBedSplit），提取 6 种序列的 embedding：
  Mut_hap1, WT_hap1, Mut_hap2, WT_hap2, Mut_ref, WT_ref

每种序列的 embedding 由两部分 concat 而成：
  emb = concat(emb_up, emb_down)

其中：
  emb_up   = upstream  正链序列 → 推理 → 取末尾 w 个 token → pooling
  emb_down = downstream 反向互补链 → 推理 → 取末尾 w 个 token → pooling

参数
----
  w        = var_len + 2（变异区域 + 紧邻上下游各 1 bp）
  var_len  = max(len(alt) - len(ref) + 1, 1)
  pooling  = "mean"（可选 "max"）

最终 embedding 维度 = hidden_size × 2

缓存策略
--------
全局哈希缓存（内存中）：
  - 键：序列哈希 + w + pooling 方法（确保不同参数下不会误命中）
  - 值：序列级 tail pooled embedding（np.ndarray）
  - 基因型等价规则保证不重复构建：wt_is_alias_of_mut 时直接复用 mut 的 embedding

使用示例
--------
    extractor = EmbeddingExtractor(
        embedding_manager=manager,
        pooling="mean",
    )

    six_embs = extractor.extract(
        bed_split=split,
        center_variant=variant,
        up_result=six_seq_result["upstream"],
        dn_result=six_seq_result["downstream"],
    )
    # six_embs["Mut_hap1"] → numpy array, shape=[hidden_size * 2]
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

from core.variant import Variant, VariantBedSplit
from core.sequence_builder import HapSeqPair, SixSeqResult, reverse_complement


# ===========================================================================
# 序列哈希工具
# ===========================================================================

def _seq_hash(seq: str) -> str:
    """计算序列内容的哈希（作为缓存键的一部分）。"""
    if _HAS_XXHASH:
        return xxhash.xxh64(seq.encode()).hexdigest()
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def _cache_key(seq: str, w: int, pooling: str) -> str:
    """生成包含序列、窗口大小和 pooling 方法的缓存键。"""
    return f"{_seq_hash(seq)}_w{w}_{pooling}"


# ===========================================================================
# 核心提取器
# ===========================================================================

class EmbeddingExtractor:
    """
    个性化 VEP embedding 提取器。

    参数
    ----
    embedding_manager : EmbeddingManager 实例（本地 GPU 模式）
    pooling           : pooling 方法，"mean"（默认）或 "max"
    context_window    : 保留参数，暂不使用（未来可调整 w 的计算方式）
    cache             : 全局共享缓存 dict。None 表示关闭全局缓存
                        （仅在同一变异内部去重，不跨变异/跨样本缓存）
    """

    # 6 种序列的键名
    SEQ_KEYS = [
        "Mut_hap1", "WT_hap1",
        "Mut_hap2", "WT_hap2",
        "Mut_ref",  "WT_ref",
    ]

    def __init__(
        self,
        embedding_manager,
        pooling: str = "mean",
        context_window: Optional[int] = None,  # 保留参数，暂不使用
        cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.manager = embedding_manager
        self.pooling = pooling
        self.context_window = context_window
        # 全局 embedding 缓存：
        #   dict 实例 → 跨变异/跨样本共享缓存（use_global_cache=True）
        #   None      → 关闭全局缓存，仅在 extract() 内部临时去重
        self._cache: Optional[Dict[str, np.ndarray]] = cache

    # -----------------------------------------------------------------------
    # 工具：计算变异区域提取窗口 w
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_w(center_variant: Variant) -> int:
        """DEPRECATED: 返回 mut 的 w。请使用 compute_w_mut / compute_w_wt。"""
        return len(center_variant.alt) + 2

    @staticmethod
    def compute_w_mut(center_variant: Variant) -> int:
        """mut 序列 tail pooling 窗口大小：alt 区域 + 上下游各 1 bp"""
        return len(center_variant.alt) + 2

    @staticmethod
    def compute_w_wt(center_variant: Variant) -> int:
        """wt 序列 tail pooling 窗口大小：ref 区域 + 上下游各 1 bp"""
        return len(center_variant.ref) + 2

    # -----------------------------------------------------------------------
    # 主接口
    # -----------------------------------------------------------------------

    def extract(
        self,
        center_variant: Variant,
        up_result: SixSeqResult,
        dn_result: SixSeqResult,
    ) -> Dict[str, np.ndarray]:
        """
        从 SixSeqResult（upstream + downstream）提取 6 种 embedding。

        参数
        ----
        center_variant : 目标变异（用于计算 w）
        up_result      : upstream 方向的 SixSeqResult
        dn_result      : downstream 方向的 SixSeqResult

        返回
        ----
        {
            "Mut_hap1": np.ndarray [hidden_size*2],
            "WT_hap1":  np.ndarray [hidden_size*2],
            "Mut_hap2": np.ndarray [hidden_size*2],
            "WT_hap2":  np.ndarray [hidden_size*2],
            "Mut_ref":  np.ndarray [hidden_size*2],
            "WT_ref":   np.ndarray [hidden_size*2],
        }
        """
        w_mut = self.compute_w_mut(center_variant)
        w_wt = self.compute_w_wt(center_variant)

        # 使用全局缓存或临时缓存（关闭全局缓存时，仍在同一变异内部去重）
        cache = self._cache if self._cache is not None else {}

        # 1. 按 w 分组收集序列（mut/wt 序列长度不同时 tail 窗口也不同）
        key_to_seq_by_w: Dict[int, Dict[str, str]] = {w_mut: {}, w_wt: {}}

        # 2. 建立 (hap_name, mut_or_wt, direction) → cache_key 的映射
        key_to_cache: Dict[Tuple[str, str, str], str] = {}

        for direction, result in [("up", up_result), ("dn", dn_result)]:
            for hap_name, pair in [
                ("hap1", result.hap1),
                ("hap2", result.hap2),
                ("ref",  result.ref_pair),
            ]:
                for mut_or_wt in ("mut", "wt"):
                    w = w_mut if mut_or_wt == "mut" else w_wt
                    if mut_or_wt == "mut":
                        seq_fwd = pair.mut
                    else:
                        seq_fwd = pair.wt if not pair.wt_is_alias_of_mut else pair.mut

                    if direction == "dn":
                        seq_for_infer = reverse_complement(seq_fwd)
                    else:
                        seq_for_infer = seq_fwd

                    ck = _cache_key(seq_for_infer, w, self.pooling)
                    key_to_seq_by_w[w][ck] = seq_for_infer
                    key_to_cache[(direction, hap_name, mut_or_wt)] = ck

        # 3. 按 w 分组批量推理（不同 w 的序列分开推理，保证 tail pooling 正确）
        for w, kt_seq in key_to_seq_by_w.items():
            seqs_to_infer = [
                (ck, seq) for ck, seq in kt_seq.items()
                if ck not in cache
            ]
            if seqs_to_infer:
                cache_keys, seqs = zip(*seqs_to_infer)
                self._run_inference(list(seqs), list(cache_keys), w, cache)

        # 4. 组装 6 种 embedding（emb_up concat emb_down）
        result: Dict[str, np.ndarray] = {}

        hap_to_key_map = {
            "Mut_hap1": ("hap1", "mut"),
            "WT_hap1":  ("hap1", "wt"),
            "Mut_hap2": ("hap2", "mut"),
            "WT_hap2":  ("hap2", "wt"),
            "Mut_ref":  ("ref",  "mut"),
            "WT_ref":   ("ref",  "wt"),
        }

        for emb_name, (hap_name, mut_or_wt) in hap_to_key_map.items():
            up_ck = key_to_cache[("up", hap_name, mut_or_wt)]
            dn_ck = key_to_cache[("dn", hap_name, mut_or_wt)]

            emb_up = cache[up_ck]
            emb_dn = cache[dn_ck]

            result[emb_name] = np.concatenate([emb_up, emb_dn], axis=0)

        return result

    def extract_batch(
        self,
        variants: List[Variant],
        up_results: List[SixSeqResult],
        dn_results: List[SixSeqResult],
    ) -> List[Dict[str, np.ndarray]]:
        """
        批量提取多个变异的 embedding（共享 cache，最大化 GPU 利用率）。

        参数
        ----
        variants   : 变异列表
        up_results : 对应的 upstream SixSeqResult 列表
        dn_results : 对应的 downstream SixSeqResult 列表

        返回
        ----
        List[Dict[str, np.ndarray]]（每个元素对应一个变异的 6 种 embedding）
        """
        # 1. 收集所有变异的所有序列，按 w 分组去重
        # cache_key → seq_str，按 w 分组存储
        per_w_seqs: Dict[int, Dict[str, str]] = {}
        per_variant_key_maps: List[Dict[Tuple[str, str, str], str]] = []

        for variant, up_r, dn_r in zip(variants, up_results, dn_results):
            w_mut = self.compute_w_mut(variant)
            w_wt = self.compute_w_wt(variant)
            for w in (w_mut, w_wt):
                if w not in per_w_seqs:
                    per_w_seqs[w] = {}

            key_to_cache: Dict[Tuple[str, str, str], str] = {}

            for direction, result in [("up", up_r), ("dn", dn_r)]:
                for hap_name, pair in [
                    ("hap1", result.hap1),
                    ("hap2", result.hap2),
                    ("ref",  result.ref_pair),
                ]:
                    for mut_or_wt in ("mut", "wt"):
                        w = w_mut if mut_or_wt == "mut" else w_wt
                        if mut_or_wt == "mut":
                            seq_fwd = pair.mut
                        else:
                            seq_fwd = pair.wt if not pair.wt_is_alias_of_mut else pair.mut

                        seq_for_infer = (
                            reverse_complement(seq_fwd)
                            if direction == "dn"
                            else seq_fwd
                        )

                        ck = _cache_key(seq_for_infer, w, self.pooling)
                        per_w_seqs[w][ck] = seq_for_infer
                        key_to_cache[(direction, hap_name, mut_or_wt)] = ck

            per_variant_key_maps.append(key_to_cache)

        # 2. 按 w 分组推理（保证每个 w 使用正确的 pooling 窗口）
        cache = self._cache if self._cache is not None else {}
        for w, ck_to_seq in per_w_seqs.items():
            seqs_to_infer = [
                (ck, seq) for ck, seq in ck_to_seq.items()
                if ck not in cache
            ]
            if seqs_to_infer:
                cache_keys, seqs = zip(*seqs_to_infer)
                self._run_inference(list(seqs), list(cache_keys), w, cache)

        # 3. 组装每个变异的 6 种 embedding
        hap_to_key_map = {
            "Mut_hap1": ("hap1", "mut"),
            "WT_hap1":  ("hap1", "wt"),
            "Mut_hap2": ("hap2", "mut"),
            "WT_hap2":  ("hap2", "wt"),
            "Mut_ref":  ("ref",  "mut"),
            "WT_ref":   ("ref",  "wt"),
        }

        results = []
        for key_to_cache in per_variant_key_maps:
            emb_dict: Dict[str, np.ndarray] = {}
            for emb_name, (hap_name, mut_or_wt) in hap_to_key_map.items():
                up_ck = key_to_cache[("up", hap_name, mut_or_wt)]
                dn_ck = key_to_cache[("dn", hap_name, mut_or_wt)]
                emb_up = cache[up_ck]
                emb_dn = cache[dn_ck]
                emb_dict[emb_name] = np.concatenate([emb_up, emb_dn], axis=0)
            results.append(emb_dict)

        return results

    # -----------------------------------------------------------------------
    # 内部：批量推理 + tail pooling → 写入缓存
    # -----------------------------------------------------------------------

    def _run_inference(
        self,
        seqs: List[str],
        cache_keys: List[str],
        w: int,
        cache: Dict[str, np.ndarray],
    ) -> None:
        """
        对 seqs 中未命中缓存的序列做批推理，tail pooling 后写入 cache。

        参数
        ----
        seqs       : 待推理序列列表
        cache_keys : 对应的缓存键列表（包含序列哈希 + w + pooling）
        w          : tail pooling 窗口大小
        cache      : 目标缓存 dict（全局或临时）
        """
        import torch

        batch_size = self.manager.batch_size

        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i: i + batch_size]
            batch_cache_keys = cache_keys[i: i + batch_size]

            # 推理
            hidden, mask = self.manager.get_hidden_states(batch_seqs)
            # hidden: [B, L, D], mask: [B, L]

            # tail pooling
            pooled = self.manager.tail_pool(hidden, mask, w=w, method=self.pooling)
            # pooled: [B, D]

            # 写入缓存
            for j, ck in enumerate(batch_cache_keys):
                if ck not in cache:
                    cache[ck] = (
                        pooled[j].float().numpy().astype("float32")
                    )

    # -----------------------------------------------------------------------
    # 缓存管理
    # -----------------------------------------------------------------------

    def cache_size(self) -> int:
        return len(self._cache) if self._cache is not None else 0

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
