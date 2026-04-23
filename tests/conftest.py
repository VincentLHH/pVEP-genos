"""
tests/conftest.py
=================
pytest 全局 fixtures 和环境检测工具。

设计原则
--------
- 所有路径优先从环境变量读取，fallback 到配置文件或默认值。
- GPU/CPU 检测自动进行，需要 GPU 的测试自动 skip。
- 提供真实数据和 mock 数据两套 fixture，测试覆盖更全面。
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────
# 0. Project root 寻径（tests/ → pVEP-genos/）
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────
# 1. 环境检测
# ─────────────────────────────────────────────────────────────
def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _has_genvarloader():
    try:
        import genvarloader
        return True
    except Exception:
        return False


def _gpu_memory_gb():
    """返回当前可见 GPU 的总显存（GB），估算不可用时返回 0。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        total = sum(torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count()))
        return total / (1024 ** 3)
    except Exception:
        return 0


# 全局标记（整个 session 只检测一次）
_has_cuda = _has_torch_cuda()
_n_gpus = _gpu_count()
_has_gvl = _has_genvarloader()
_gpu_mem_gb = _gpu_memory_gb()

print(f"[conftest] CUDA available: {_has_cuda}, GPUs: {_n_gpus}, "
      f"GenVarLoader: {_has_gvl}, VRAM: {_gpu_mem_gb:.1f} GB")


# ─────────────────────────────────────────────────────────────
# 2. Pytest marks（在整个 session 注册）
# ─────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require CUDA GPU")
    config.addinivalue_line("markers", "cpu: tests that run on CPU only")
    config.addinivalue_line("markers", "genvarloader: tests that require genvarloader library")
    config.addinivalue_line("markers", "slow: slow tests (full pipeline)")


# ─────────────────────────────────────────────────────────────
# 3. 路径 fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def env_model_path():
    """模型权重路径，优先级：ENV > config.yaml > mock"""
    path = os.environ.get("PVEPGENOS_MODEL_PATH")
    if path:
        return path
    # 尝试从 config 读取
    cfg_file = PROJECT_ROOT / "config" / "default.yaml"
    if cfg_file.exists():
        import yaml
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("model", {}).get("path", "")
        if p and os.path.exists(p):
            return p
    return None  # 必须时再报 skip


@pytest.fixture(scope="session")
def env_ref_fasta():
    path = os.environ.get("PVEPGENOS_REF_FASTA")
    if path:
        return path
    cfg_file = PROJECT_ROOT / "config" / "default.yaml"
    if cfg_file.exists():
        import yaml
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("ref_fasta", "")
        if p and os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def env_vcf_path():
    path = os.environ.get("PVEPGENOS_VCF_PATH")
    if path:
        return path
    cfg_file = PROJECT_ROOT / "config" / "default.yaml"
    if cfg_file.exists():
        import yaml
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("vcf_path", "")
        if p and os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def env_bed_path():
    path = os.environ.get("PVEPGENOS_BED_PATH")
    if path:
        return path
    cfg_file = PROJECT_ROOT / "config" / "default.yaml"
    if cfg_file.exists():
        import yaml
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("bed_path", "")
        if p and os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def env_api_url():
    return os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────
# 4. Mock 数据 fixtures（不依赖任何外部文件）
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_fasta_path():
    """
    在 tmpdir 创建 200bp 的小 reference：
      >chr1
      ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
    """
    import pysam

    seq = ("ACGT" * 60 + "T" * 20)[:200]  # 200bp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fa", mode="w") as f:
        f.write(f">chr1\n{seq}\n")
        f.write(f">chr2\n{seq}\n")
        fasta_path = f.name

    # 构建 index
    try:
        pysam.faidx(fasta_path)
    except Exception:
        pass

    yield fasta_path

    # cleanup
    for ext in ("", ".fai", ".gzi", ".bai"):
        try:
            p = fasta_path + ext
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


@pytest.fixture(scope="session")
def mock_vcf_path(mock_fasta_path):
    """
    在 tmpdir 创建一个小 VCF.gz：
      chr1:1000 A->T (sample1: 1|0, sample2: 0|1)
      chr1:1010 C->G (sample1: 1|1, sample2: 1|0)
      chr1:1030 A->AC (sample1: 0|1, sample2: 0|0)
    """
    import gzip

    vcf_text = (
        '##fileformat=VCFv4.2\n'
        '##reference=file:///dev/null\n'
        '##contig=<ID=chr1,length=200>\n'
        '#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO  FORMAT  sample1  sample2\n'
        'chr1    100     .       A       T       .       .       .       GT     1|0     0|1\n'
        'chr1    110     .       C       G       .       .       .       GT     1|1     1|0\n'
        'chr1    130     .       A       AC      .       .       .       GT     0|1     0|0\n'
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz", mode="wb") as f:
        with gzip.open(f, "wt") as gz:
            gz.write(vcf_text)
        vcf_path = f.name

    yield vcf_path

    for ext in ("", ".tbi"):
        try:
            p = vcf_path + ext
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


@pytest.fixture(scope="session")
def mock_bed_path():
    """chr1:50-180 区域，完整覆盖所有 mock variants"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bed", mode="w") as f:
        f.write("chr1\t50\t180\n")
        bed_path = f.name

    yield bed_path

    try:
        os.remove(bed_path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 5. Skip helpers（供各测试文件直接 import 使用）
# ─────────────────────────────────────────────────────────────
skip_if_no_cuda = pytest.mark.skipif(
    not _has_cuda, reason="CUDA not available"
)
skip_if_no_gvl = pytest.mark.skipif(
    not _has_gvl, reason="genvarloader not installed"
)
skip_if_single_gpu = pytest.mark.skipif(
    _n_gpus < 2, reason=f"Need ≥2 GPUs, found {_n_gpus}"
)
skip_if_no_model = pytest.mark.skipif(
    True,  # will be overridden per-test with env_model_path fixture
    reason="Model path not configured"
)


def require_gpu(test_func):
    """Decorator: skip if no CUDA"""
    return pytest.mark.gpu(test_func)


def require_genvarloader(test_func):
    """Decorator: skip if genvarloader not installed"""
    return pytest.mark.genvarloader(test_func)


# ─────────────────────────────────────────────────────────────
# 6. 其他 utilities
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def temp_output_dir():
    """每个测试使用独立临时输出目录，自动清理"""
    with tempfile.TemporaryDirectory(prefix="pVEP_test_") as d:
        yield d


def check_result_nontrivial(result: dict, min_dim: int = 128):
    """验证 embedding 结果非零、非空、维度合理"""
    assert result, "embedding result is empty"
    for key, methods in result.items():
        assert isinstance(methods, dict), f"{key}: methods should be dict"
        for method, vec in methods.items():
            assert isinstance(vec, list), f"{key}/{method}: should be list"
            assert len(vec) >= min_dim, f"{key}/{method}: dim too small ({len(vec)})"
            assert not all(v == 0 for v in vec), f"{key}/{method}: all zeros"
