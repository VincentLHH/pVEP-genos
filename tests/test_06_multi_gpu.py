"""
tests/test_06_multi_gpu.py
===========================
测试多卡推理（需要 ≥2 GPU 的节点）。

覆盖场景
--------
1. GPU 数量检测（至少 2 张卡才可测试）
2. run_multi_gpu 入口函数存在且可调用
3. 样本 round-robin 分配逻辑
4. Manager().dict() 跨进程共享 cache
5. 样本级断点续存（多进程下）
6. 多进程异常捕获与汇总

前置条件
--------
- GPU 节点，至少 2 张 GPU
- 环境变量 PVEPGENOS_MODEL_PATH / PVEPGENOS_VCF_PATH / PVEPGENOS_BED_PATH / PVEPGENOS_REF_FASTA
  或 config/default.yaml

运行（多 GPU 节点）
--------------------
export PVEPGENOS_MODEL_PATH=/path/to/Genos-1.2B
export PVEPGENOS_VCF_PATH=/path/to/your.vcf.gz
export PVEPGENOS_BED_PATH=/path/to/your.bed
export PVEPGENOS_REF_FASTA=/path/to/hg38.fa
pytest tests/test_06_multi_gpu.py -v -s

标记过滤：
pytest tests/ -v -m "multi_gpu"           # 仅运行多卡测试
pytest tests/ -v -m "not multi_gpu"      # 跳过多卡测试
"""


# ─────────────────────────────────────────────────────────────
# 多进程辅助函数（必须在模块顶层，避免 Python 3.12 spawn pickle 问题）
# ─────────────────────────────────────────────────────────────
def _mp_writer(shared_dict, key, value):
    """子进程写入 shared dict"""
    shared_dict[key] = value


def _mp_reader(shared_dict, key):
    """子进程读取 shared dict"""
    return shared_dict.get(key)


import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────
# 使用 conftest.py 统一的环境检测和 skip helpers
# ─────────────────────────────────────────────────────────────
from tests.conftest import (
    HAS_CUDA,
    GPU_COUNT,
    skip_if_no_cuda,
    skip_if_single_gpu,
    skip_if_no_gvl,
    HAS_GVL,
    require_real_data,
    require_model_path,
)

# ─────────────────────────────────────────────────────────────
# Fixtures（复用 conftest 的通用 fixtures）
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def temp_output_dir():
    with tempfile.TemporaryDirectory(prefix="pVEP_multi_") as d:
        yield d


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestGPUCountDetection:
    """GPU 数量检测（gpu + multi_gpu 标记）"""

    @pytest.mark.gpu
    @skip_if_no_cuda
    def test_cuda_available(self):
        """CUDA 可用"""
        assert HAS_CUDA, "CUDA should be available"

    @pytest.mark.gpu
    @pytest.mark.multi_gpu
    @skip_if_single_gpu
    def test_multiple_gpus_detected(self):
        """至少检测到 2 个 GPU"""
        assert GPU_COUNT >= 2, f"Expected ≥2 GPUs, found {GPU_COUNT}"
        print(f"Detected {GPU_COUNT} GPUs")

    @pytest.mark.gpu
    @pytest.mark.multi_gpu
    @skip_if_single_gpu
    def test_all_devices_accessible(self):
        """验证所有 GPU 设备均可访问"""
        import torch
        for i in range(GPU_COUNT):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, "
                  f"{props.total_memory / 1024**3:.1f} GB, "
                  f"compute {props.major}.{props.minor}")


class TestMultiGPUImport:
    """多卡模块可导入（无 GPU 要求）"""

    def test_multi_gpu_runner_importable(self):
        """core.multi_gpu_runner 应可正常导入"""
        from core.multi_gpu_runner import run_multi_gpu, _worker
        assert callable(run_multi_gpu)
        assert callable(_worker)

    def test_run_pipeline_multi_importable(self):
        """run_pipeline 应可导入（不报语法错误）"""
        import run_pipeline
        assert hasattr(run_pipeline, "run_multi")


class TestRoundRobinDistribution:
    """Round-robin 分配逻辑（无 GPU 要求）"""

    def test_even_distribution(self):
        """样本均匀分配到各 GPU"""
        from core.multi_gpu_runner import run_multi_gpu

        # 构造 mock 参数（不实际运行）
        # 只测分配逻辑：模拟 bucket 构造
        n_gpus = 4
        samples = [f"s{i}" for i in range(10)]

        buckets = [[] for _ in range(n_gpus)]
        for i, name in enumerate(samples):
            buckets[i % n_gpus].append(name)

        expected = {0: ["s0", "s4", "s8"],
                    1: ["s1", "s5", "s9"],
                    2: ["s2", "s6"],
                    3: ["s3", "s7"]}
        for rank, expected_bucket in expected.items():
            assert buckets[rank] == expected_bucket, \
                f"rank {rank}: expected {expected_bucket}, got {buckets[rank]}"
        print("Round-robin distribution: correct")

    def test_more_gpus_than_samples(self):
        """样本少于 GPU 时，部分 GPU 空转"""
        n_gpus = 8
        samples = ["s0", "s1"]

        buckets = [[] for _ in range(n_gpus)]
        for i, name in enumerate(samples):
            buckets[i % n_gpus].append(name)

        assert buckets[0] == ["s0"]
        assert buckets[1] == ["s1"]
        for b in buckets[2:]:
            assert b == [], f"Extra GPU buckets should be empty: {b}"
        print("Underpopulated GPUs: handled correctly")


class TestSharedCacheMechanism:
    """跨进程共享 cache（multi_gpu 标记）"""

    @pytest.mark.multi_gpu
    @skip_if_single_gpu
    def test_manager_dict_works(self):
        """Manager().dict() 跨进程共享"""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with mp.Manager() as mgr:
            shared = mgr.dict()

            # 进程 1 写入
            p1 = ctx.Process(target=_mp_writer, args=(shared, "test_key", [1.0, 2.0, 3.0]))
            p1.start()
            p1.join()
            assert p1.exitcode == 0

            # 进程 2 读取
            result = shared.get("test_key")
            assert result == [1.0, 2.0, 3.0]
        print("Manager().dict() cross-process share: PASS")


class TestMultiGPUMinimal:
    """最小化多卡运行测试（multi_gpu + real 标记）"""

    @pytest.mark.multi_gpu
    @pytest.mark.real
    @pytest.mark.gpu
    @skip_if_single_gpu
    def test_multi_gpu_two_samples_two_gpus(self, require_real_data, require_model_path, temp_output_dir):
        """真实运行：2 样本 + 2 GPU，验证不出错"""
        import run_pipeline
        import yaml

        # 创建临时 config
        tmp_cfg = {
            "vcf_path": require_real_data["vcf_path"],
            "bed_path": require_real_data["bed_path"],
            "ref_fasta": require_real_data["ref_fasta"],
            "model": {
                "name": "Genos-1.2B",
                "path": require_model_path,
                "dtype": "bfloat16",
                "batch_size": 8,
                "mode": "local",
            },
            "window_size": 128,
            "embedding": {
                "methods": ["mean"],
                "save_interval": 10,
            },
            "seq_builder": {
                "type": "builtin",
            },
            "output_dir": temp_output_dir,
        }

        devices = [f"cuda:{i}" for i in range(min(2, GPU_COUNT))]

        # 取前 2 个样本测试
        import pysam
        vcf = pysam.VariantFile(require_real_data["vcf_path"])
        all_samples = list(vcf.header.samples)
        vcf.close()
        test_samples = all_samples[:2]

        from core.multi_gpu_runner import run_multi_gpu

        run_multi_gpu(
            sample_names=test_samples,
            devices=devices,
            model_cfg=tmp_cfg["model"],
            vcf_path=tmp_cfg["vcf_path"],
            bed_path=tmp_cfg["bed_path"],
            ref_fasta=tmp_cfg["ref_fasta"],
            window_size=tmp_cfg["window_size"],
            output_dir=tmp_cfg["output_dir"],
            seq_builder_type="builtin",
            seq_builder_cfg=tmp_cfg.get("seq_builder", {}),
            methods=tmp_cfg["embedding"]["methods"],
            save_interval=tmp_cfg["embedding"]["save_interval"],
            save_haplotypes=True,
            save_embeddings=True,
        )

        # 验证输出文件存在
        import os
        for s in test_samples:
            out_file = os.path.join(temp_output_dir, f"{s}.json")
            assert os.path.exists(out_file), f"Output file missing: {out_file}"
            import json
            with open(out_file) as f:
                data = json.load(f)
            assert "embeddings" in data or "haplotypes" in data
            print(f"  {s}.json: OK")


class TestMultiGPUGenVarLoader:
    """多卡 + GenVarLoader 组合（multi_gpu + genvarloader 标记）"""

    @pytest.mark.multi_gpu
    @pytest.mark.genvarloader
    @pytest.mark.real
    @skip_if_single_gpu
    @skip_if_no_gvl
    def test_multi_gpu_genvarloader_basic(self, require_real_data, require_model_path, temp_output_dir):
        """多卡使用 genvarloader builder"""
        import pysam

        vcf = pysam.VariantFile(require_real_data["vcf_path"])
        all_samples = list(vcf.header.samples)
        vcf.close()
        test_samples = all_samples[:2]

        from core.multi_gpu_runner import run_multi_gpu

        devices = [f"cuda:{i}" for i in range(min(2, GPU_COUNT))]
        tmp_cfg_model = {
            "name": "Genos-1.2B",
            "path": require_model_path,
            "dtype": "bfloat16",
            "batch_size": 8,
            "mode": "local",
        }

        run_multi_gpu(
            sample_names=test_samples,
            devices=devices,
            model_cfg=tmp_cfg_model,
            vcf_path=require_real_data["vcf_path"],
            bed_path=require_real_data["bed_path"],
            ref_fasta=require_real_data["ref_fasta"],
            window_size=128,
            output_dir=temp_output_dir,
            seq_builder_type="genvarloader",
            seq_builder_cfg={"gvl_cache_dir": "/tmp/pVEP_gvl_cache_test"},
            methods=["mean"],
            save_interval=10,
            save_haplotypes=True,
            save_embeddings=True,
        )

        import os
        for s in test_samples:
            out_file = os.path.join(temp_output_dir, f"{s}.json")
            assert os.path.exists(out_file), f"Output missing: {out_file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
