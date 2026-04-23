import os
import json
from typing import List, Dict, Optional


class Sample:
    """
    单个样本的状态管理与推理协调。

    参数
    ----
    save_haplotypes : bool
        是否在 JSON 中保存重建的单倍型序列（可能很大，默认 True）。
    save_embeddings : bool
        是否进行推理并保存 embedding（若为 False 则跳过模型推理，默认 True）。
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
            return False  # 不保存 embedding 时始终重新处理（一般不会走到这里）
        return (
            variant_id in self.embeddings
            and model_name in self.embeddings[variant_id]
        )

    # =========================================================
    # 🔥 主处理逻辑（真正 batch 推理版本）
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
        批量处理所有 variant：
        1. 先对所有 variant 做序列构建（CPU）；
        2. 将全部待推理序列一次性送入 EmbeddingManager.bulk_get_embeddings（GPU）；
        3. 结果写回并定期保存。

        与旧版逐样本逐 variant 调用 get_embeddings 的区别：
        - 旧版：每个 variant 各自凑一小批，GPU 利用率低。
        - 新版：所有 variant 的序列在一次 bulk_get_embeddings 中统一推理，
          内部自动去重 + 按 batch_size 分块，GPU 吞吐量最大化。
        """
        model_name = embedding_manager.model_name

        # -------- 阶段 1：构建序列（纯 CPU） --------
        print(f"[{self.sample_id}] 🧬 Building sequences for {len(variants)} variants...")

        # variant_id → seq_data（包含 hap1/hap2 的各种链）
        seq_data_map: Dict[str, Optional[dict]] = {}

        # 扁平化字典：{flat_key: sequence_str}，用于批推理
        # flat_key 格式："{variant_id}||{hap}_{seq_type}"
        flat_seq_dict: Dict[str, str] = {}

        skipped = 0
        for v in variants:
            vid = v.id

            # 已完整处理过，跳过
            if self.save_embeddings and self.is_processed(vid, model_name):
                skipped += 1
                continue

            seq_data = sequence_builder.build(v, variants)
            if seq_data is None:
                seq_data_map[vid] = None
                continue

            seq_data_map[vid] = seq_data

            # 保存单倍型（若开启）
            if self.save_haplotypes:
                self.haplotypes[vid] = seq_data

            # 只有需要 embedding 才加入推理队列
            if self.save_embeddings:
                for hap in ("hap1", "hap2"):
                    if hap not in seq_data:
                        continue
                    for key, seq in seq_data[hap].items():
                        flat_key = f"{vid}||{hap}_{key}"
                        flat_seq_dict[flat_key] = seq

        if skipped:
            print(f"[{self.sample_id}] ⏭  Skipped {skipped} already-processed variants")

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

                # 还原 {hap_key: {method: vec}} 结构
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

        self.save()
        print(f"[{self.sample_id}] ✅ Done")
