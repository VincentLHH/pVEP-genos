import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional

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

    接口
    ----
    bulk_get_embeddings(seq_dict, methods) → {key: {method: [float]}}
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
    # pooling（仅 local 模式使用）
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
    # 本地推理核心（仅 local 模式）
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
    # 🔥 统一入口
    # =========================================================
    def bulk_get_embeddings(
        self,
        seq_dict: Dict[str, str],
        methods: List[str] = ["mean"],
    ) -> Dict[str, Dict[str, list]]:
        """
        跨 variant / 跨样本的统一推理入口。
        local  模式：本地 GPU 推理，支持 cache 命中。
        api    模式：HTTP 请求远程服务（服务端维护 cache）。

        输入
        ----
        seq_dict : {flat_key: sequence_str}

        返回
        ----
        {flat_key: {method: [float, ...]}}
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
