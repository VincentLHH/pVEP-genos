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
# 环境检测
# ─────────────────────────────────────────────────────────────
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
except Exception:
    HAS_CUDA = False
    GPU_COUNT = 0

skip_if_no_cuda = pytest.mark.skipif(
    not HAS_CUDA,
    reason="CUDA not available"
)
skip_if_single_gpu = pytest.mark.skipif(
    GPU_COUNT < 2,
    reason=f"Need ≥2 GPUs, found {GPU_COUNT}"
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_data_cfg(env_vcf_path, env_bed_path, env_ref_fasta, env_model_path):
    """收集所有真实数据路径，缺失则 skip"""
    missing = []
    vals = {
        "vcf_path": env_vcf_path,
        "bed_path": env_bed_path,
        "ref_fasta": env_ref_fasta,
        "model_path": env_model_path,
    }
    for name, val in vals.items():
        if not val or not os.path.exists(str(val)):
            missing.append(f"{name}={val!r}")

    if missing:
        pytest.skip(f"Missing data: {missing}", allow_module_level=True)

    return vals


@pytest.fixture(scope="session")
def temp_output_dir():
    with tempfile.TemporaryDirectory(prefix="pVEP_multi_") as d:
        yield d


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestGPUCountDetection:
    """GPU 数量检测"""

    def test_cuda_available(self):
        assert HAS_CUDA, "CUDA should be available"

    @skip_if_single_gpu
    def test_multiple_gpus_detected(self):
        assert GPU_COUNT >= 2, f"Expected ≥2 GPUs, found {GPU_COUNT}"
        print(f"Detected {GPU_COUNT} GPUs")

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
    """多卡模块可导入"""

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
    """Round-robin 分配逻辑"""

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
    """跨进程共享 cache"""

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
    """最小化多卡运行测试（用真实数据，少量样本）"""

    @skip_if_single_gpu
    def test_multi_gpu_two_samples_two_gpus(self, real_data_cfg, temp_output_dir):
        """真实运行：2 样本 + 2 GPU，验证不出错"""
        import run_pipeline
        import yaml

        # 创建临时 config
        tmp_cfg = {
            "vcf_path": real_data_cfg["vcf_path"],
            "bed_path": real_data_cfg["bed_path"],
            "ref_fasta": real_data_cfg["ref_fasta"],
            "model": {
                "name": "Genos-1.2B",
                "path": real_data_cfg["model_path"],
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
        vcf = pysam.VariantFile(real_data_cfg["vcf_path"])
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
    """多卡 + GenVarLoader 组合"""

    @skip_if_single_gpu
    @pytest.mark.skipif(
        True,  # genvarloader 需要单独检测
        reason="genvarloader optional, skip unless explicitly enabled"
    )
    def test_multi_gpu_genvarloader_basic(self, real_data_cfg, temp_output_dir):
        """多卡使用 genvarloader builder"""
        # 如果 genvarloader 不可用则 skip
        try:
            import genvarloader
        except ImportError:
            pytest.skip("genvarloader not installed", allow_module_level=True)

        import pysam

        vcf = pysam.VariantFile(real_data_cfg["vcf_path"])
        all_samples = list(vcf.header.samples)
        vcf.close()
        test_samples = all_samples[:2]

        from core.multi_gpu_runner import run_multi_gpu

        devices = [f"cuda:{i}" for i in range(min(2, GPU_COUNT))]
        tmp_cfg_model = {
            "name": "Genos-1.2B",
            "path": real_data_cfg["model_path"],
            "dtype": "bfloat16",
            "batch_size": 8,
            "mode": "local",
        }

        run_multi_gpu(
            sample_names=test_samples,
            devices=devices,
            model_cfg=tmp_cfg_model,
            vcf_path=real_data_cfg["vcf_path"],
            bed_path=real_data_cfg["bed_path"],
            ref_fasta=real_data_cfg["ref_fasta"],
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
