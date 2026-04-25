"""
multi_gpu_runner.py
===================
多卡并行推理协调器。

设计原则
--------
- 每张 GPU 独占一个子进程（避免 CUDA 上下文竞争）。
- 样本按 round-robin 分配给各 GPU。
- Cache 用 multiprocessing.Manager().dict() 全局共享：
    key  = (sequence_str, method_str)
    value= list[float]
  任一进程推理完毕后写入共享 cache，其他进程下次遇到相同序列
  可直接命中，无需重复推理。
- 共享 cache 的写入通过 Manager 代理对象序列化，线程安全，
  但存在一定序列化开销；对于大批量序列，net 收益仍然显著。

注意
----
- 需要在 __main__ guard 下调用（multiprocessing 在 Windows 需要 spawn）。
- 模型权重在每个子进程中独立加载（每张卡各一份），这是正常行为。
"""

import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import List, Dict, Optional
import traceback

from core.sample import Sample
from core.sequence_builder import SequenceBuilder
from core.genvarloader_builder import GenVarLoaderSequenceBuilder
from models.embedding_manager import EmbeddingManager


# =========================================================
# 子进程工作函数
# =========================================================
def _worker(
    rank: int,
    sample_names: List[str],
    device: str,
    model_cfg: dict,
    vcf_path: str,
    bed_path: str,
    regions: List[tuple],
    ref_fasta: str,
    window_size: int,
    output_dir: str,
    seq_builder_type: str,
    seq_builder_cfg: dict,
    pooling: str,
    save_interval: int,
    save_haplotypes: bool,
    save_embeddings: bool,
    shared_cache: DictProxy,
    bed_split_n: int = 200,
):
    """
    在单张 GPU 上顺序处理分配给本进程的全部样本。
    """
    import pysam  # 在子进程内导入，避免 fork 问题
    from core.embedding_extractor import EmbeddingExtractor

    print(f"[GPU {rank} / {device}] 🚀 Worker started, {len(sample_names)} samples")

    try:
        # ----- 构建 sequence builder -----
        if seq_builder_type == "genvarloader":
            builder = GenVarLoaderSequenceBuilder(
                vcf_path=vcf_path,
                bed_path=bed_path,
                ref_fasta=ref_fasta,
                gvl_cache_dir=seq_builder_cfg.get("gvl_cache_dir", "/tmp/gvl_cache"),
                strandaware=seq_builder_cfg.get("gvl_strandaware", True),
                max_mem=seq_builder_cfg.get("gvl_max_mem", "4g"),
            )
            print(f"[GPU {rank}]    🔧 Using GenVarLoaderSequenceBuilder")
        else:
            builder = SequenceBuilder(ref_fasta, window_size=window_size)
            print(f"[GPU {rank}]    🔧 Using built-in SequenceBuilder")

        # ----- 加载模型（注入共享 cache）-----
        manager_obj = EmbeddingManager(
            model_name=model_cfg["name"],
            model_path=model_cfg["path"],
            device=device,
            dtype=model_cfg.get("dtype", "bfloat16"),
            batch_size=model_cfg["batch_size"],
            shared_cache=shared_cache,  # ← 注入全局共享 cache
            mode=model_cfg.get("mode", "local"),
        )

        extractor = EmbeddingExtractor(
            embedding_manager=manager_obj,
            pooling=pooling,
        )

        # ----- 打开 VCF（每个子进程独立 file handle）-----
        vcf = pysam.VariantFile(vcf_path)

        for sample_name in sample_names:
            sample = Sample(
                sample_name,
                output_dir,
                save_haplotypes=save_haplotypes,
                save_embeddings=save_embeddings,
            )

            # 样本级断点续存
            if sample.is_complete([], model_cfg["name"]):
                print(f"[GPU {rank}] ⏭  [{sample_name}] 完整结果已存在，跳过")
                continue

            print(f"[GPU {rank}] 📂 Processing {sample_name}")
            variants = _load_variants(vcf, regions, sample_name)
            print(f"[GPU {rank}]    🧬 {len(variants)} variants")

            sample.process_all_v2(
                variants,
                builder,
                extractor,
                n=bed_split_n,
                save_interval=save_interval,
            )

        vcf.close()
        print(f"[GPU {rank} / {device}] ✅ Worker done")

    except Exception:
        print(f"[GPU {rank} / {device}] ❌ Worker crashed:")
        traceback.print_exc()
        raise


def _load_bed(bed_path: str) -> List[tuple]:
    """加载 BED 文件，返回 [(chrom, start, end), ...] 列表（0-indexed, half-open）。"""
    regions = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            regions.append((chrom, start, end))
    return regions


def _load_variants(vcf, regions, sample_name):
    """从 VCF 中提取指定样本在 regions 内的非 ref 变异。"""
    from core.variant import Variant

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


# =========================================================
# 公共入口：多卡并行
# =========================================================
def run_multi_gpu(
    sample_names: List[str],
    devices: List[str],
    model_cfg: dict,
    vcf_path: str,
    bed_path: str,
    ref_fasta: str,
    window_size: int,
    output_dir: str,
    seq_builder_type: str,
    seq_builder_cfg: dict,
    pooling: str,
    save_interval: int,
    save_haplotypes: bool,
    save_embeddings: bool,
    bed_split_n: int = 200,
):
    """
    将 sample_names 按 round-robin 分配给 devices，
    每个设备起一个子进程并行处理。

    共享 cache 通过 Manager().dict() 实现跨进程共享。
    """
    n = len(devices)

    # 加载 BED regions
    regions = _load_bed(bed_path)
    print(f"📋 Loaded {len(regions)} regions from {bed_path}")

    # 按 round-robin 分桶
    buckets: List[List[str]] = [[] for _ in range(n)]
    for i, name in enumerate(sample_names):
        buckets[i % n].append(name)

    print(f"🗂  Sample distribution across {n} GPU(s):")
    for i, (dev, bucket) in enumerate(zip(devices, buckets)):
        print(f"   [{dev}] rank={i} → {len(bucket)} samples")

    # 创建共享 cache（Manager 进程代理）
    ctx = mp.get_context("spawn")  # Windows / CUDA 安全
    with mp.Manager() as mgr:
        shared_cache = mgr.dict()
        print("🔗 Shared cache initialized")

        processes = []
        for rank, (device, bucket) in enumerate(zip(devices, buckets)):
            if not bucket:
                continue

            p = ctx.Process(
                target=_worker,
                kwargs=dict(
                    rank=rank,
                    sample_names=bucket,
                    device=device,
                    model_cfg=model_cfg,
                    vcf_path=vcf_path,
                    bed_path=bed_path,
                    regions=regions,
                    ref_fasta=ref_fasta,
                    window_size=window_size,
                    output_dir=output_dir,
                    seq_builder_type=seq_builder_type,
                    seq_builder_cfg=seq_builder_cfg,
                    pooling=pooling,
                    save_interval=save_interval,
                    save_haplotypes=save_haplotypes,
                    save_embeddings=save_embeddings,
                    shared_cache=shared_cache,
                    bed_split_n=bed_split_n,
                ),
                name=f"worker-{device}",
            )
            p.start()
            processes.append(p)

        # 等待所有子进程完成
        failed = []
        for p in processes:
            p.join()
            if p.exitcode != 0:
                failed.append(p.name)

        if failed:
            raise RuntimeError(
                f"❌ The following worker(s) failed: {failed}"
            )

        print(f"✅ All {len(processes)} GPU worker(s) finished")
        print(f"   Shared cache final size: {len(shared_cache)} entries")
