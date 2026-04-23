"""
run_pipeline.py
===============
pVEP-genos 主入口。

新增功能
--------
1. 真正的 batch 推理（sample.process_all 内部统一收集序列再批推理）。
2. --save-haplotypes / --no-save-haplotypes  控制是否保存单倍型序列。
   --save-embeddings  / --no-save-embeddings 控制是否进行推理保存 embedding。
3. 多卡推理：--devices cuda:0 cuda:1 ...
   各 GPU 独占一个子进程，全局 cache 跨进程共享（Manager.dict）。

用法示例
--------
# 单卡，默认保存一切
python run_pipeline.py --config config/default.yaml

# 多卡，不保存单倍型序列（节省磁盘），只保存 embedding
python run_pipeline.py --config config/default.yaml \\
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \\
    --no-save-haplotypes

# 只构建序列，不推理（用于 debug 序列构建）
python run_pipeline.py --config config/default.yaml --no-save-embeddings
"""

import yaml
import argparse
import pysam
from tqdm import tqdm

from core.sample import Sample
from core.variant import Variant
from core.sequence_builder import SequenceBuilder
from models.embedding_manager import EmbeddingManager


# =========================
# 🔹 CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="pVEP-genos embedding pipeline"
    )
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")

    # 多卡覆盖（优先级高于 config 中的 devices 字段）
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help="使用的 GPU 列表，例如 --devices cuda:0 cuda:1（覆盖 config）",
    )

    # 保存控制（--no-* 形式显式关闭）
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
# 🔹 resolve 运行时参数
# =========================
def resolve_flags(args, cfg):
    """
    命令行参数优先级 > config 文件 > 默认值（True）。
    """
    output_cfg = cfg.get("output", {})

    save_haplotypes = (
        args.save_haplotypes
        if args.save_haplotypes is not None
        else output_cfg.get("save_haplotypes", True)
    )
    save_embeddings = (
        args.save_embeddings
        if args.save_embeddings is not None
        else output_cfg.get("save_embeddings", True)
    )
    return save_haplotypes, save_embeddings


def resolve_devices(args, cfg):
    """
    返回要使用的设备列表。
    命令行 --devices 优先级最高；其次 config.model.devices；
    最后 fallback 到 config.model.device（单卡）。
    """
    if args.devices:
        return args.devices

    devices_in_cfg = cfg["model"].get("devices", [])
    if devices_in_cfg:
        return devices_in_cfg

    return [cfg["model"].get("device", "cuda")]


# =========================
# 🔹 单卡主流程
# =========================
def run_single(cfg, devices, save_haplotypes, save_embeddings):
    device = devices[0]
    print(f"▶️  Single-GPU mode on {device}")

    vcf = pysam.VariantFile(cfg["vcf_path"])
    samples = list(vcf.header.samples)
    regions = load_bed(cfg["bed_path"])

    builder = SequenceBuilder(cfg["ref_fasta"], window_size=cfg["window_size"])

    manager = EmbeddingManager(
        model_name=cfg["model"]["name"],
        model_path=cfg["model"]["path"],
        device=device,
        dtype=cfg["model"].get("dtype", "bfloat16"),
        batch_size=cfg["model"]["batch_size"],
    )

    for sample_name in tqdm(samples, desc="Samples"):
        print(f"\n🚀 Processing {sample_name}")

        sample = Sample(
            sample_name,
            cfg["output_dir"],
            save_haplotypes=save_haplotypes,
            save_embeddings=save_embeddings,
        )

        variants = load_variants(vcf, regions, sample_name)
        print(f"   🧬 {len(variants)} variants")

        sample.process_all(
            variants,
            builder,
            manager,
            methods=cfg["embedding"]["methods"],
            save_interval=cfg["embedding"]["save_interval"],
        )


# =========================
# 🔹 多卡主流程
# =========================
def run_multi(cfg, devices, save_haplotypes, save_embeddings):
    from core.multi_gpu_runner import run_multi_gpu

    print(f"▶️  Multi-GPU mode on {devices}")

    vcf = pysam.VariantFile(cfg["vcf_path"])
    samples = list(vcf.header.samples)
    vcf.close()

    regions = load_bed(cfg["bed_path"])

    run_multi_gpu(
        sample_names=samples,
        devices=devices,
        model_cfg=cfg["model"],
        vcf_path=cfg["vcf_path"],
        regions=regions,
        ref_fasta=cfg["ref_fasta"],
        window_size=cfg["window_size"],
        output_dir=cfg["output_dir"],
        methods=cfg["embedding"]["methods"],
        save_interval=cfg["embedding"]["save_interval"],
        save_haplotypes=save_haplotypes,
        save_embeddings=save_embeddings,
    )


# =========================
# 🔹 main
# =========================
def main():
    args = parse_args()
    cfg = load_config(args.config)

    save_haplotypes, save_embeddings = resolve_flags(args, cfg)
    devices = resolve_devices(args, cfg)

    print(f"📋 Config:")
    print(f"   save_haplotypes = {save_haplotypes}")
    print(f"   save_embeddings = {save_embeddings}")
    print(f"   devices         = {devices}")

    if len(devices) > 1:
        run_multi(cfg, devices, save_haplotypes, save_embeddings)
    else:
        run_single(cfg, devices, save_haplotypes, save_embeddings)


if __name__ == "__main__":
    main()
