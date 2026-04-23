import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
from collections import defaultdict


class EmbeddingManager:
    """
    单卡 embedding 推理器。

    改进点
    ------
    1. cache 可由外部注入（multiprocessing.Manager().dict()），
       支持多卡进程间全局共享。
    2. 新增 bulk_get_embeddings()：接受 {任意key: seq} 的大字典，
       一次性去重 + batch 推理，不再逐样本调用。
    3. 原 get_embeddings() 保持兼容，内部走同一 _run_batch_inference。
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 8,
        shared_cache: Optional[dict] = None,   # ← 多卡共享 cache 注入口
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        # 🔥 全局 cache：key=(seq, method) → list[float]
        #    若外部注入则用共享 dict，否则进程本地 dict
        self.cache: dict = shared_cache if shared_cache is not None else {}

        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        print(f"🔄 Loading model [{model_name}] on {device} from {model_path}...")
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

    # =========================================================
    # pooling
    # =========================================================
    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor, method: str):
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

    # =========================================================
    # 底层 tokenize + 前向
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
    # 核心：对一组去重序列做 batch 推理，结果写入 cache
    # =========================================================
    def _run_batch_inference(
        self,
        unique_seqs: List[str],
        methods: List[str],
    ) -> None:
        """
        对 unique_seqs 中尚未命中 cache 的序列做 batch 推理，
        推理结果写入 self.cache。
        """
        # 过滤掉已经全部命中的序列
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
    # 🔥 新接口：bulk 批推理
    #    输入: {key: seq, ...}（可跨多个 variant，量级可以很大）
    #    输出: {key: {method: [float, ...]}, ...}
    # =========================================================
    def bulk_get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, list]]:
        """
        跨 variant / 跨样本的统一 batch 推理入口。
        先全局去重，再一次性送入 GPU，最后按 key 重新分发结果。
        """
        if not seq_dict:
            return {}

        # 1️⃣ 收集所有独立序列（去重）
        unique_seqs = list(dict.fromkeys(seq_dict.values()))  # 保序去重

        # 2️⃣ 统一推理（只推尚未命中 cache 的）
        self._run_batch_inference(unique_seqs, methods)

        # 3️⃣ 按 key 组装结果
        result: Dict[str, Dict[str, list]] = {}
        for key, seq in seq_dict.items():
            result[key] = {}
            for method in methods:
                result[key][method] = self.cache[(seq, method)]

        return result

    # =========================================================
    # 兼容旧接口（逐次调用，内部也走 bulk 路径）
    # =========================================================
    def get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, list]]:
        return self.bulk_get_embeddings(seq_dict, methods)
