import os
import json
from typing import List, Dict, Optional

import numpy as np


class Sample:
    """
    单个样本的状态管理与推理协调。

    参数
    ----
    save_haplotypes : bool
        是否在输出中保存重建的单倍型序列（可能很大，默认 True）。
    save_embeddings : bool
        是否进行推理并保存 embedding（若为 False 则跳过模型推理，默认 True）。

    新版 Embedding 格式
    -------------------
    embeddings[variant_id][model_name] = {
        "Mut_hap1": [float, ...],   # 维度 = hidden_size * 2
        "WT_hap1":  [float, ...],
        "Mut_hap2": [float, ...],
        "WT_hap2":  [float, ...],
        "Mut_ref":  [float, ...],
        "WT_ref":   [float, ...],
    }
    """

    def __init__(
        self,
        sample_id: str,
        output_dir: str,
        save_haplotypes: bool = True,
        save_embeddings: bool = True,
    ):
        self.sample_id = sample_id
        self.output_dir = output_dir
        self.save_haplotypes = save_haplotypes
        self.save_embeddings = save_embeddings

        self.haplotypes: Dict = {}
        self.embeddings: Dict = {}

        os.makedirs(self.output_dir, exist_ok=True)
        self.filepath = os.path.join(self.output_dir, f"{sample_id}.json")

        self._load_if_exists()

    # =========================================================
    # Load / Save
    # =========================================================
    def _load_if_exists(self):
        if os.path.exists(self.filepath):
            print(f"🔁 Resuming sample {self.sample_id}")
            with open(self.filepath, "r") as f:
                data = json.load(f)
                self.haplotypes = data.get("haplotypes", {})
                self.embeddings = data.get("embeddings", {})

    def save(self):
        data: Dict = {"sample_id": self.sample_id}

        if self.save_haplotypes:
            data["haplotypes"] = self.haplotypes

        if self.save_embeddings:
            data["embeddings"] = self.embeddings

        with open(self.filepath, "w") as f:
            json.dump(data, f)

    # =========================================================
    # 判断是否已处理
    # =========================================================
    def is_processed(self, variant_id: str, model_name: str) -> bool:
        if not self.save_embeddings:
            return False
        return (
            variant_id in self.embeddings
            and model_name in self.embeddings[variant_id]
        )

    def is_complete(self, variant_ids: list, model_name: str) -> bool:
        """
        样本级完整性检查：
        - JSON 存在
        - 包含所有传入 variant_id 的 embedding（允许部分 variant 无 embedding，如构建失败）
        - 至少有一个 variant 的 embedding 存在（确保不是空文件）
        只有在满足上述条件时，才认为该样本"完整"，可直接跳过整个处理流程。
        """
        if not self.save_embeddings:
            return False
        if not os.path.exists(self.filepath):
            return False
        if not self.embeddings:
            return False
        # 至少有一个 variant 的 embedding 存在，才算"有效文件"
        return any(model_name in emb_dict for emb_dict in self.embeddings.values())

    # =========================================================
    # 🔥 新版主处理逻辑（基于 EmbeddingExtractor + 6 种序列）
    # =========================================================
    def process_all_v2(
        self,
        variants: list,
        sequence_builder,
        embedding_extractor,
        n: int = 200,
        save_interval: int = 50,
    ):
        """
        新版批量处理所有 variant（6 种序列 + tail pooling + concat）。

        步骤
        ----
        1. 对每个 variant 构建 VariantBedSplit（BED 拆分）；
        2. 调用 sequence_builder.build_six_seqs() 构建 6 种序列；
        3. 调用 embedding_extractor.extract() 提取 6 种 embedding；
        4. 定期保存。

        参数
        ----
        variants          : Variant 列表
        sequence_builder  : SequenceBuilder 或 GenVarLoaderSequenceBuilder 实例
        embedding_extractor : EmbeddingExtractor 实例
        n                 : BED 拆分侧翼长度
        save_interval     : 每处理多少个 variant 保存一次
        """
        from core.variant import split_variant_to_bed, VariantBedSplit

        model_name = embedding_extractor.manager.model_name

        # ── 注入当前样本名（GenVarLoaderSequenceBuilder 需要此信息）──
        if hasattr(sequence_builder, "current_sample"):
            sequence_builder.current_sample = self.sample_id

        skipped = 0
        built = 0
        count = 0

        print(f"[{self.sample_id}] 🧬 Processing {len(variants)} variants (new v2 pipeline)...")

        for v in variants:
            vid = v.id

            # 已完整处理过，跳过
            if self.save_embeddings and self.is_processed(vid, model_name):
                skipped += 1
                continue

            # 1. BED 拆分
            try:
                bed_split = split_variant_to_bed(v, n=n)
            except ValueError as e:
                print(f"[{self.sample_id}] ⚠ BED split failed for {vid}: {e}")
                continue

            # 2. 构建 6 种序列（兼容 builtin 和 genvarloader 两种 builder）
            is_gvl = hasattr(sequence_builder, "build_six_seqs") and hasattr(
                sequence_builder, "vcf_path"
            )

            if is_gvl:
                # GenVarLoader 模式
                six_seqs = sequence_builder.build_six_seqs(
                    bed_split=bed_split,
                    center_variant=v,
                    sample_name=self.sample_id,
                )
            else:
                # builtin 模式
                six_seqs = sequence_builder.build_six_seqs(
                    bed_split=bed_split,
                    center_variant=v,
                    variants_in_region=variants,  # 传入全量 variants 用于背景过滤
                )

            if six_seqs is None:
                print(f"[{self.sample_id}] ⚠ Sequence build failed for {vid}")
                continue

            built += 1

            # 保存单倍型（若开启）
            if self.save_haplotypes:
                self.haplotypes[vid] = self._serialize_six_seqs(six_seqs)

            # 3. 提取 embedding
            if self.save_embeddings:
                try:
                    emb_dict = embedding_extractor.extract(
                        center_variant=v,
                        up_result=six_seqs["upstream"],
                        dn_result=six_seqs["downstream"],
                    )
                except Exception as e:
                    print(f"[{self.sample_id}] ⚠ Embedding extraction failed for {vid}: {e}")
                    continue

                if vid not in self.embeddings:
                    self.embeddings[vid] = {}

                # 将 numpy array 转为可 JSON 序列化的 list
                self.embeddings[vid][model_name] = {
                    k: v_emb.tolist() for k, v_emb in emb_dict.items()
                }

            count += 1
            if count % save_interval == 0:
                print(f"[{self.sample_id}] 💾 Saving at {count} variants "
                      f"(cache size: {embedding_extractor.cache_size()})")
                self.save()

        if skipped:
            print(f"[{self.sample_id}] ⏭  Skipped {skipped} already-processed variants")
        print(f"[{self.sample_id}]    Built {built} variant sequences")

        # 恢复 builder 属性（避免跨样本污染）
        if hasattr(sequence_builder, "current_sample"):
            sequence_builder.current_sample = None

        self.save()
        print(f"[{self.sample_id}] ✅ Done")

    # =========================================================
    # 🔥 旧版主处理逻辑（全序列 pooling，保留兼容）
    # =========================================================
    def process_all(
        self,
        variants: list,
        sequence_builder,
        embedding_manager,
        methods: List[str] = ["mean"],
        save_interval: int = 50,
    ):
        """
        [旧版] 批量处理所有 variant（全序列 pooling）。
        新代码请使用 process_all_v2()。

        批量处理所有 variant：
        1. 先对所有 variant 做序列构建（CPU）；
        2. 将全部待推理序列一次性送入 EmbeddingManager.bulk_get_embeddings（GPU）；
        3. 结果写回并定期保存。

        GenVarLoader 支持：
        - 底层 builder 注入 current_sample 供 genvarloader 查询样本列；
        - 同一 region 的多个 variant 共享序列（通过 variant_id 区分 key）。
        """
        model_name = embedding_manager.model_name

        # ── 注入当前样本名（GenVarLoaderSequenceBuilder 需要此信息）──
        if hasattr(sequence_builder, "current_sample"):
            sequence_builder.current_sample = self.sample_id

        # -------- 阶段 1：构建序列（纯 CPU） --------
        print(f"[{self.sample_id}] 🧬 Building sequences for {len(variants)} variants...")

        seq_data_map: Dict[str, Optional[dict]] = {}
        flat_seq_dict: Dict[str, str] = {}

        skipped = 0
        built = 0

        for v in variants:
            vid = v.id

            if self.save_embeddings and self.is_processed(vid, model_name):
                skipped += 1
                continue

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                seq_data = sequence_builder.build(v, variants)

            if seq_data is None:
                seq_data_map[vid] = None
                continue

            built += 1
            seq_data_map[vid] = seq_data

            if self.save_haplotypes:
                self.haplotypes[vid] = seq_data

            if self.save_embeddings:
                for hap in ("hap1", "hap2"):
                    if hap not in seq_data:
                        continue
                    for key, seq in seq_data[hap].items():
                        flat_key = f"{vid}||{hap}_{key}"
                        flat_seq_dict[flat_key] = seq

        if skipped:
            print(f"[{self.sample_id}] ⏭  Skipped {skipped} already-processed variants")
        print(f"[{self.sample_id}]    Built {built} variant sequences")

        # -------- 阶段 2：批推理（GPU） --------
        if self.save_embeddings and flat_seq_dict:
            print(
                f"[{self.sample_id}] 🚀 Running bulk inference on "
                f"{len(flat_seq_dict)} sequences "
                f"({len(set(flat_seq_dict.values()))} unique after dedup)..."
            )
            flat_emb = embedding_manager.bulk_get_embeddings(flat_seq_dict, methods)
        else:
            flat_emb = {}

        # -------- 阶段 3：写回 + 定期保存 --------
        if self.save_embeddings:
            count = 0
            for vid, seq_data in seq_data_map.items():
                if seq_data is None:
                    continue

                if vid not in self.embeddings:
                    self.embeddings[vid] = {}

                per_variant_emb: Dict[str, Dict[str, list]] = {}
                for hap in ("hap1", "hap2"):
                    if hap not in seq_data:
                        continue
                    for key in seq_data[hap]:
                        flat_key = f"{vid}||{hap}_{key}"
                        if flat_key in flat_emb:
                            per_variant_emb[f"{hap}_{key}"] = flat_emb[flat_key]

                self.embeddings[vid][model_name] = per_variant_emb

                count += 1
                if count % save_interval == 0:
                    print(f"[{self.sample_id}] 💾 Saving at {count} variants")
                    self.save()

        # 恢复 builder 属性（避免跨样本污染）
        if hasattr(sequence_builder, "current_sample"):
            sequence_builder.current_sample = None

        self.save()
        print(f"[{self.sample_id}] ✅ Done")

    # =========================================================
    # 序列化辅助
    # =========================================================

    def _serialize_six_seqs(self, six_seqs: dict) -> dict:
        """将 SixSeqResult 字典序列化为 JSON 可存储格式。"""
        result = {}
        for direction, sr in six_seqs.items():
            result[direction] = {
                "hap1": {
                    "mut": sr.hap1.mut,
                    "wt":  sr.hap1.wt,
                    "wt_is_alias": sr.hap1.wt_is_alias_of_mut,
                },
                "hap2": {
                    "mut": sr.hap2.mut,
                    "wt":  sr.hap2.wt,
                    "wt_is_alias": sr.hap2.wt_is_alias_of_mut,
                },
                "ref": {
                    "mut": sr.ref_pair.mut,
                    "wt":  sr.ref_pair.wt,
                    "wt_is_alias": sr.ref_pair.wt_is_alias_of_mut,
                },
            }
        return result
