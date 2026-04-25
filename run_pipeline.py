"""
run_pipeline.py
===============
pVEP-genos 主入口。

功能
----
1. 真正的 batch 推理（sample.process_all 内部统一收集序列再批推理）。
2. --save-haplotypes / --no-save-haplotypes  控制是否保存单倍型序列。
   --save-embeddings  / --no-save-embeddings 控制是否进行推理保存 embedding。
3. 多卡推理：--devices cuda:0 cuda:1 ...
   各 GPU 独占一个子进程，全局 cache 跨进程共享（Manager.dict）。
4. API 推理模式：--mode api --api-base-url http://gpu-node:8000
   无 GPU 节点通过 HTTP 调用远程 GPU 服务器。

用法示例
--------
# 单卡，默认保存一切
python run_pipeline.py --config config/default.yaml

# 多卡，不保存单倍型序列
python run_pipeline.py --config config/default.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --no-save-haplotypes

# 只构建序列，不推理（debug）
python run_pipeline.py --config config/default.yaml --no-save-embeddings

# API 模式（无 GPU 节点，调用远程服务）
python run_pipeline.py --config config/default.yaml \
    --mode api --api-base-url http://192.168.1.100:8000
"""

import yaml
import argparse
import pysam
from tqdm import tqdm

from core.sample import Sample
from core.variant import Variant
from core.sequence_builder import SequenceBuilder
from core.genvarloader_builder import GenVarLoaderSequenceBuilder
from core.embedding_extractor import EmbeddingExtractor
from models.embedding_manager import EmbeddingManager


# =========================
# 🔹 CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="pVEP-genos embedding pipeline"
    )
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")

    # ── 推理模式 ──
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default=None,
        dest="mode",
        help="推理方式：local=本机 GPU，api=调用远程 API 服务（默认由 config 决定）",
    )
    parser.add_argument(
        "--api-base-url",
        dest="api_base_url",
        default=None,
        help="API 服务地址，例如 http://192.168.1.100:8000（mode=api 时必须）",
    )

    # ── 序列构建器 ──
    parser.add_argument(
        "--seq-builder",
        choices=["builtin", "genvarloader"],
        default=None,
        dest="seq_builder",
        help="序列构建方式：builtin=内置 Python 实现，genvarloader=GenVarLoader C扩展（更快，默认由 config 决定）",
    )

    # ── 多卡覆盖（仅 local 模式有效）──
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help="使用的 GPU 列表，例如 --devices cuda:0 cuda:1（覆盖 config，仅 local 模式）",
    )

    # ── 保存控制 ──
    parser.add_argument(
        "--save-haplotypes",
        dest="save_haplotypes",
        action="store_true",
        default=None,
        help="在 JSON 中保存重建的单倍型序列（默认由 config 决定）",
    )
    parser.add_argument(
        "--no-save-haplotypes",
        dest="save_haplotypes",
        action="store_false",
        help="不保存单倍型序列",
    )
    parser.add_argument(
        "--save-embeddings",
        dest="save_embeddings",
        action="store_true",
        default=None,
        help="进行推理并保存 embedding（默认由 config 决定）",
    )
    parser.add_argument(
        "--no-save-embeddings",
        dest="save_embeddings",
        action="store_false",
        help="跳过推理，不保存 embedding",
    )

    return parser.parse_args()


# =========================
# 🔹 load config
# =========================
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =========================
# 🔹 BED
# =========================
def load_bed(bed_path):
    regions = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            chrom, start, end = line.strip().split()[:3]
            regions.append((chrom, int(start), int(end)))
    return regions


# =========================
# 🔹 VCF → Variant
# =========================
def load_variants(vcf, regions, sample_name):
    variants = []
    for chrom, start, end in regions:
        for rec in vcf.fetch(chrom, start, end):
            if not rec.alts:
                continue

            alt = rec.alts[0]
            ref = rec.ref
            gt = rec.samples[sample_name]["GT"]

            if gt is None or None in gt:
                continue
            if sum(gt) == 0:
                continue

            variants.append(Variant(chrom, rec.pos, ref, alt, gt))

    return variants


# =========================
# 🔹 resolve 参数
# =========================
def resolve_mode(args, cfg) -> str:
    mode = args.mode if args.mode else cfg.get("model", {}).get("mode", "local")
    if mode not in ("local", "api"):
        raise ValueError(f"mode must be 'local' or 'api', got {mode!r}")
    return mode


def resolve_api_base_url(args, cfg, mode: str) -> str:
    if mode == "api":
        url = args.api_base_url or cfg.get("model", {}).get("api_base_url")
        if not url:
            raise ValueError(
                "API mode requires --api-base-url or config.model.api_base_url"
            )
        return url
    return ""


def resolve_flags(args, cfg):
    """统一解析 save_haplotypes / save_embeddings，优先 CLI > embedding > output（兼容旧版）"""
    emb_cfg = cfg.get("embedding", {})
    out_cfg = cfg.get("output", {})

    save_haplotypes = (
        args.save_haplotypes
        if args.save_haplotypes is not None
        else emb_cfg.get("save_haplotypes", out_cfg.get("save_haplotypes", True))
    )
    save_embeddings = (
        args.save_embeddings
        if args.save_embeddings is not None
        else emb_cfg.get("save_embeddings", out_cfg.get("save_embeddings", True))
    )
    return save_haplotypes, save_embeddings


def resolve_seq_builder_type(args, cfg) -> str:
    """返回 'builtin' 或 'genvarloader'。"""
    if args.seq_builder:
        return args.seq_builder
    return cfg.get("seq_builder", {}).get("type", "builtin")


def make_builder(cfg, seq_builder_type: str):
    """
    工厂函数：根据 seq_builder_type 创建对应的序列构建器。
    """
    if seq_builder_type == "genvarloader":
        gvl_cfg = cfg.get("seq_builder", {})
        gvl_cache_dir = gvl_cfg.get("gvl_cache_dir", "/tmp/gvl_cache")
        print(f"🔧 Using GenVarLoaderSequenceBuilder (cache={gvl_cache_dir})")
        return GenVarLoaderSequenceBuilder(
            vcf_path=cfg["vcf_path"],
            bed_path=cfg["bed_path"],
            ref_fasta=cfg["ref_fasta"],
            gvl_cache_dir=gvl_cache_dir or "/tmp/gvl_cache",
            strandaware=gvl_cfg.get("gvl_strandaware", True),
            max_mem=gvl_cfg.get("gvl_max_mem", "4g"),
        )
    else:
        print("🔧 Using built-in SequenceBuilder")
        return SequenceBuilder(
            cfg["ref_fasta"],
            window_size=cfg["window_size"]
        )


def resolve_devices(args, cfg):
    if args.devices:
        return args.devices
    devices_in_cfg = cfg["model"].get("devices", [])
    if devices_in_cfg:
        return devices_in_cfg
    return [cfg["model"].get("device", "cuda")]


# =========================
# 🔹 EmbeddingManager 工厂
# =========================
def make_manager(cfg, device: str, api_base_url: str = ""):
    common = dict(
        model_name=cfg["model"]["name"],
        batch_size=cfg["model"]["batch_size"],
    )
    if api_base_url:
        return EmbeddingManager(
            **common,
            model_path="",         # API 模式不需要本地模型路径
            device="cpu",
            mode="api",
            api_base_url=api_base_url,
        )
    else:
        return EmbeddingManager(
            **common,
            model_path=cfg["model"]["path"],
            device=device,
            dtype=cfg["model"].get("dtype", "bfloat16"),
            mode="local",
        )


# =========================
# 🔹 单卡主流程（local 或 api 模式共用）
# =========================
def run_single(cfg, devices, seq_builder_type, mode, api_base_url, save_haplotypes, save_embeddings):
    device = devices[0]
    mode_label = "API" if mode == "api" else f"Single-GPU on {device}"
    print(f"▶️  Mode: {mode_label}")

    vcf = pysam.VariantFile(cfg["vcf_path"])
    samples = list(vcf.header.samples)
    regions = load_bed(cfg["bed_path"])

    builder = make_builder(cfg, seq_builder_type)
    manager = make_manager(cfg, device, api_base_url)

    embedding_cfg = cfg.get("embedding", {})
    pooling = embedding_cfg.get("pooling", "mean")
    n = cfg.get("bed_split", {}).get("n", 200)
    save_interval = embedding_cfg.get("save_interval", 50)
    use_global_cache = embedding_cfg.get("use_global_cache", True)
    variant_batch_size = embedding_cfg.get("variant_batch_size", 16)

    # 缓存：use_global_cache=True 时使用全局 dict，False 时设为 None（由 Extractor 内部处理）
    cache = {} if use_global_cache else None

    extractor = EmbeddingExtractor(
        embedding_manager=manager,
        pooling=pooling,
        context_window=embedding_cfg.get("context_window"),
        cache=cache,
    )

    for sample_name in tqdm(samples, desc="Samples"):
        sample = Sample(
            sample_name,
            cfg["output_dir"],
            save_haplotypes=save_haplotypes,
            save_embeddings=save_embeddings,
        )

        # 样本级断点续存（CPU 优化，跳过已完整处理的样本）
        if sample.is_complete([], manager.model_name):
            print(f"\n🚀 [{sample_name}] ⏭  完整结果已存在，跳过")
            continue

        variants = load_variants(vcf, regions, sample_name)
        print(f"\n🚀 Processing {sample_name} | {len(variants)} variants")

        sample.process_all_v2(
            variants,
            builder,
            extractor,
            n=n,
            save_interval=save_interval,
            variant_batch_size=variant_batch_size,
        )


# =========================
# 🔹 多卡主流程（仅 local 模式）
# =========================
def run_multi(cfg, devices, seq_builder_type, save_haplotypes, save_embeddings):
    from core.multi_gpu_runner import run_multi_gpu

    print(f"▶️  Multi-GPU mode on {devices}")

    vcf = pysam.VariantFile(cfg["vcf_path"])
    samples = list(vcf.header.samples)
    vcf.close()

    regions = load_bed(cfg["bed_path"])

    embedding_cfg = cfg.get("embedding", {})
    pooling = embedding_cfg.get("pooling", "mean")
    n = cfg.get("bed_split", {}).get("n", 200)
    use_global_cache = embedding_cfg.get("use_global_cache", True)
    variant_batch_size = embedding_cfg.get("variant_batch_size", 16)

    run_multi_gpu(
        sample_names=samples,
        devices=devices,
        model_cfg=cfg["model"],
        vcf_path=cfg["vcf_path"],
        bed_path=cfg["bed_path"],
        ref_fasta=cfg["ref_fasta"],
        window_size=cfg["window_size"],
        output_dir=cfg["output_dir"],
        seq_builder_type=seq_builder_type,
        seq_builder_cfg=cfg.get("seq_builder", {}),
        pooling=pooling,
        save_interval=embedding_cfg.get("save_interval", 50),
        save_haplotypes=save_haplotypes,
        save_embeddings=save_embeddings,
        bed_split_n=n,
        use_global_cache=use_global_cache,
        variant_batch_size=variant_batch_size,
    )


# =========================
# 🔹 main
# =========================
def main():
    args = parse_args()
    cfg = load_config(args.config)

    seq_builder_type = resolve_seq_builder_type(args, cfg)
    mode = resolve_mode(args, cfg)
    api_base_url = resolve_api_base_url(args, cfg, mode)
    save_haplotypes, save_embeddings = resolve_flags(args, cfg)
    devices = resolve_devices(args, cfg)

    print("📋 Config:")
    print(f"   seq_builder      = {seq_builder_type}")
    print(f"   mode             = {mode}")
    if mode == "api":
        print(f"   api_base_url     = {api_base_url}")
    else:
        print(f"   devices          = {devices}")
    print(f"   save_haplotypes  = {save_haplotypes}")
    print(f"   save_embeddings  = {save_embeddings}")

    # API 模式走单卡路径（推理在远端）
    if mode == "api":
        run_single(cfg, ["cpu"], seq_builder_type, mode, api_base_url, save_haplotypes, save_embeddings)
    elif len(devices) > 1:
        run_multi(cfg, devices, seq_builder_type, save_haplotypes, save_embeddings)
    else:
        run_single(cfg, devices, seq_builder_type, mode, api_base_url, save_haplotypes, save_embeddings)


if __name__ == "__main__":
    main()
