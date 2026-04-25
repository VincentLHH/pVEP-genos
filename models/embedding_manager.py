import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple

import numpy as np

from api.client import EmbeddingAPIClient


class EmbeddingManager:
    """
    Embedding 推理管理器，支持两种运行模式：

    mode = "local"（默认）
        在本机 GPU 上推理（torch + transformers）。
        cache 存储在进程本地 dict（或外部传入的共享 dict）。

    mode = "api"
        通过 HTTP 调用远程推理服务（GPU 节点部署的 api.service）。
        cache 完全由服务端管理，本地只做透传。
        适合从无 GPU 节点调用 GPU 服务器。

    新版接口
    --------
    get_hidden_states(sequences) → [B, L, D] 逐 token hidden state（仅 local 模式）
    tail_pool(hidden, w, method) → [B, D] 对最后 w 个 token pooling
    bulk_get_embeddings(seq_dict, methods) → {key: {method: [float]}}  (旧接口，全序列 pooling)

    两种模式接口完全对齐，调用方无感知。
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 8,
        shared_cache: Optional[dict] = None,
        # ──────── API 模式扩展参数 ────────
        mode: str = "local",
        api_base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.mode = mode
        self.api_base_url = api_base_url

        # 本地 GPU 模式：初始化模型
        if self.mode == "local":
            self.cache: dict = shared_cache if shared_cache is not None else {}

            if dtype == "bfloat16":
                self.torch_dtype = torch.bfloat16
            elif dtype == "float16":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32

            print(
                f"🔄 Loading model [{model_name}] on {device} "
                f"from {model_path}..."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                attn_implementation=(
                    "flash_attention_2" if device.startswith("cuda") else None
                ),
            ).to(device)

            self.model.eval()
            print(f"✅ Model [{model_name}] loaded on {device}")

        # API 模式：初始化 HTTP 客户端
        elif self.mode == "api":
            if not api_base_url:
                raise ValueError("api_base_url is required when mode='api'")
            self._api_client = EmbeddingAPIClient(base_url=api_base_url)
            self.cache: dict = {}   # API 模式 cache 由服务端管理
            print(f"🌐 EmbeddingManager (API mode) → {api_base_url}")

        else:
            raise ValueError(f"Unknown mode: {mode!r}, expected 'local' or 'api'")

    # =========================================================
    # pooling 工具（仅 local 模式使用）
    # =========================================================

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
        """
        对整条序列做 pooling。
        hidden : [B, L, D]
        mask   : [B, L]
        返回   : [B, D]
        """
        if method == "last_token":
            seq_len = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, seq_len, :]

        elif method == "mean":
            mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask_exp).sum(dim=1)
            denom = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            return summed / denom

        elif method == "max":
            mask_exp = mask.unsqueeze(-1).expand(hidden.size()).bool()
            masked = hidden.clone()
            masked[~mask_exp] = -torch.inf
            return masked.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def tail_pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        w: int,
        method: str = "mean",
    ) -> torch.Tensor:
        """
        仅对每条序列的最后 w 个有效 token 做 pooling。

        参数
        ----
        hidden : [B, L, D] 逐 token hidden state
        mask   : [B, L] attention mask（1=有效，0=padding）
        w      : 取最后 w 个有效 token
        method : "mean" 或 "max"

        返回
        ----
        [B, D] pooled embedding

        设计说明
        --------
        Genos 是 decoder-only 生成式模型，对未来掩码。
        upstream   行正链输入，变异区域在序列末尾，
        downstream 行取反向互补后输入，变异区域同样在序列末尾。
        因此统一取末尾 w 个有效 token 进行 pooling。
        """
        B, L, D = hidden.size()

        results = []
        for b in range(B):
            # 找到该样本有效 token 的下标
            valid_indices = mask[b].nonzero(as_tuple=True)[0]
            n_valid = valid_indices.size(0)

            # 取最后 w 个（若不足 w 则取全部有效 token）
            tail_count = min(w, n_valid)
            tail_indices = valid_indices[-tail_count:]  # [tail_count]

            tail_hidden = hidden[b, tail_indices, :]    # [tail_count, D]

            if method == "mean":
                pooled = tail_hidden.mean(dim=0)        # [D]
            elif method == "max":
                pooled = tail_hidden.max(dim=0).values  # [D]
            else:
                raise ValueError(f"Unknown pooling method: {method}")

            results.append(pooled)

        return torch.stack(results, dim=0)  # [B, D]

    # =========================================================
    # 🆕 逐 token hidden state 接口（仅 local 模式）
    # =========================================================

    def get_hidden_states(
        self,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对输入序列做 tokenize + 前向，返回逐 token hidden state。

        参数
        ----
        sequences : List[str]，输入序列列表

        返回
        ----
        (hidden_states, attention_mask)
        hidden_states : [B, L, D] float32 tensor（已移到 CPU）
        attention_mask: [B, L] int64 tensor（已移到 CPU）

        注意：此接口不使用 cache，调用方负责缓存管理。
        仅支持 local 模式。
        """
        if self.mode != "local":
            raise NotImplementedError(
                "get_hidden_states() is only supported in local mode. "
                "For API mode, use bulk_get_embeddings()."
            )

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state.to(torch.float32).cpu()
        mask = inputs["attention_mask"].cpu()

        return hidden, mask

    def get_hidden_states_batched(
        self,
        sequences: List[str],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        分批次获取 hidden states，避免一次性 OOM。

        返回
        ----
        (list_of_hidden, list_of_mask)
        每个元素对应一个 batch 的 [B_i, L_i, D]
        """
        all_hidden = []
        all_mask = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i: i + self.batch_size]
            h, m = self.get_hidden_states(batch)
            all_hidden.append(h)
            all_mask.append(m)
        return all_hidden, all_mask

    # =========================================================
    # 底层 tokenize + 前向（仅 local 模式使用）
    # =========================================================

    def _encode_batch(self, sequences: List[str]):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state, inputs["attention_mask"]

    # =========================================================
    # 本地推理核心（全序列 pooling，仅 local 模式）
    # =========================================================

    def _run_batch_inference(
        self,
        unique_seqs: List[str],
        methods: List[str],
    ) -> None:
        """对尚未命中 cache 的序列做 batch 推理，结果写入 self.cache。"""
        to_infer = [
            s for s in unique_seqs
            if any((s, m) not in self.cache for m in methods)
        ]

        if not to_infer:
            return

        for i in range(0, len(to_infer), self.batch_size):
            batch_seqs = to_infer[i : i + self.batch_size]
            hidden, mask = self._encode_batch(batch_seqs)

            for method in methods:
                pooled = self._pool(hidden, mask, method)

                for j, seq in enumerate(batch_seqs):
                    cache_key = (seq, method)
                    if cache_key not in self.cache:
                        vec = (
                            pooled[j]
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                            .astype("float32")
                            .tolist()
                        )
                        self.cache[cache_key] = vec

    # =========================================================
    # 🔥 统一入口（全序列 pooling，向后兼容）
    # =========================================================

    def bulk_get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, list]]:
        """
        跨 variant / 跨样本的统一推理入口（全序列 pooling）。
        local  模式：本地 GPU 推理，支持 cache 命中。
        api    模式：HTTP 请求远程服务（服务端维护 cache）。

        输入
        ----
        seq_dict : {flat_key: sequence_str}

        返回
        ----
        {flat_key: {method: [float, ...]}}

        注意
        ----
        此接口做全序列 pooling，适合旧版兼容。
        新版个性化 VEP embedding 请使用 EmbeddingExtractor（core/embedding_extractor.py），
        它会调用 get_hidden_states() + tail_pool() 而不是此接口。
        """
        if not seq_dict:
            return {}

        if self.mode == "local":
            return self._bulk_local(seq_dict, methods)
        else:
            return self._bulk_api(seq_dict, methods)

    # =========================================================
    # local 推理路径
    # =========================================================

    def _bulk_local(
        self,
        seq_dict: Dict[str, str],
        methods: List[str],
    ) -> Dict[str, Dict[str, list]]:
        unique_seqs = list(dict.fromkeys(seq_dict.values()))
        self._run_batch_inference(unique_seqs, methods)

        result: Dict[str, Dict[str, list]] = {}
        for key, seq in seq_dict.items():
            result[key] = {}
            for method in methods:
                result[key][method] = self.cache[(seq, method)]

        return result

    # =========================================================
    # API 推理路径
    # =========================================================

    def _bulk_api(
        self,
        seq_dict: Dict[str, str],
        methods: List[str],
    ) -> Dict[str, Dict[str, list]]:
        return self._api_client.bulk_get_embeddings(seq_dict, methods)

    # =========================================================
    # 兼容旧接口
    # =========================================================

    def get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, list]]:
        return self.bulk_get_embeddings(seq_dict, methods)
