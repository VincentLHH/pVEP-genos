"""
tests/conftest.py
=================
pytest 全局 fixtures 和环境检测工具。

设计原则
--------
- 所有路径优先从环境变量读取，fallback 到配置文件或默认值。
- GPU/CUDA/GenVarLoader 检测集中在一处，避免各测试文件重复。
- 提供真实数据和 mock 数据两套 fixture，测试覆盖更全面。
- Skip helpers 统一导出，各测试文件统一 import 使用。

使用方式
--------
在测试文件中：

    from tests.conftest import (
        skip_if_no_cuda,
        skip_if_single_gpu,
        skip_if_no_gvl,
        require_gpu,
        require_genvarloader,
        HAS_CUDA,
        GPU_COUNT,
        HAS_GVL,
    )

    @skip_if_no_cuda
    def test_gpu_feature():
        ...

    @skip_if_single_gpu
    def test_multi_gpu():
        ...
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
# 1. 环境检测（集中管理，全局共享）
# ─────────────────────────────────────────────────────────────
def _has_torch_cuda():
    """检测 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_count():
    """返回 GPU 数量，估算不可用时返回 0"""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _has_genvarloader():
    """检测 genvarloader 库是否可导入"""
    try:
        import genvarloader
        return True
    except Exception:
        return False


def _gpu_memory_gb():
    """返回当前可见 GPU 的总显存（GB），估算不可用时返回 0"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        total = sum(torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count()))
        return total / (1024 ** 3)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────
# 2. 全局常量（session 级别检测，一次计算全局共享）
# ─────────────────────────────────────────────────────────────
HAS_CUDA: bool = _has_torch_cuda()
GPU_COUNT: int = _gpu_count()
HAS_GVL: bool = _has_genvarloader()
GPU_MEM_GB: float = _gpu_memory_gb()

# 打印环境信息（session 初始化时）
print(f"[conftest] Environment: "
      f"CUDA={HAS_CUDA}, GPUs={GPU_COUNT}, "
      f"GenVarLoader={HAS_GVL}, VRAM={GPU_MEM_GB:.1f} GB")


# ─────────────────────────────────────────────────────────────
# 3. Pytest marks（在整个 session 注册）
# ─────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require CUDA GPU")
    config.addinivalue_line("markers", "cpu: tests that run on CPU only")
    config.addinivalue_line("markers", "genvarloader: tests that require genvarloader library")
    config.addinivalue_line("markers", "multi_gpu: tests that require ≥2 GPUs")
    config.addinivalue_line("markers", "slow: slow tests (full pipeline)")
    config.addinivalue_line("markers", "api: tests that require API service running")
    config.addinivalue_line("markers", "real: tests that require real data files")


# ─────────────────────────────────────────────────────────────
# 4. Skip helpers（统一导出，避免各测试文件重复定义）
# ─────────────────────────────────────────────────────────────

# GPU 相关
skip_if_no_cuda = pytest.mark.skipif(
    not HAS_CUDA,
    reason=f"CUDA not available (found {GPU_COUNT} GPU(s))"
)

skip_if_single_gpu = pytest.mark.skipif(
    GPU_COUNT < 2,
    reason=f"Need ≥2 GPUs for multi-GPU test, found {GPU_COUNT}"
)

skip_if_no_model = pytest.mark.skipif(
    True,  # will be overridden per-test with env_model_path fixture
    reason="Model path not configured"
)

# GenVarLoader 相关
skip_if_no_gvl = pytest.mark.skipif(
    not HAS_GVL,
    reason="genvarloader not installed or not importable"
)

# API 服务相关（通过 fixture 检测，不单独定义 skip）
# skip_if_no_api = pytest.mark.skipif(
#     False,  # 通过 fixture api_reachable 动态检测
#     reason="API service not running"
# )


# ─────────────────────────────────────────────────────────────
# 5. Decorator helpers（可选，更语义化的写法）
# ─────────────────────────────────────────────────────────────

def require_gpu(test_func):
    """Decorator: 标记并 skip 无 GPU 的测试"""
    return pytest.mark.gpu(pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")(test_func))


def require_genvarloader(test_func):
    """Decorator: 标记并 skip 无 genvarloader 的测试"""
    return pytest.mark.genvarloader(
        pytest.mark.skipif(not HAS_GVL, reason="genvarloader not installed")(test_func)
    )


def require_multi_gpu(test_func):
    """Decorator: 标记并 skip 单 GPU 的测试"""
    return pytest.mark.multi_gpu(
        pytest.mark.skipif(GPU_COUNT < 2, reason=f"Need ≥2 GPUs, found {GPU_COUNT}")(test_func)
    )


# ─────────────────────────────────────────────────────────────
# 6. 路径 fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def env_model_path():
    """模型权重路径，优先级：ENV > config.yaml > None"""
    path = os.environ.get("PVEPGENOS_MODEL_PATH")
    if path:
        return path
    cfg_file = PROJECT_ROOT / "config" / "default.yaml"
    if cfg_file.exists():
        import yaml
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("model", {}).get("path", "")
        if p and os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def env_ref_fasta():
    """参考基因组 FASTA 路径"""
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
    """VCF 文件路径"""
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
    """BED 文件路径"""
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
    """API 服务地址"""
    return os.environ.get("PVEPGENOS_API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────
# 7. Mock 数据 fixtures（不依赖任何外部文件）
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
      chr1:100 A->T (sample1: 1|0, sample2: 0|1)
      chr1:110 C->G (sample1: 1|1, sample2: 1|0)
      chr1:130 A->AC (sample1: 0|1, sample2: 0|0)
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
# 8. 其他 utilities
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


# ─────────────────────────────────────────────────────────────
# 9. Fixture helpers（通用 skip 逻辑）
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def require_model_path(env_model_path):
    """模型路径不可用时 skip 测试"""
    if not env_model_path:
        pytest.skip("Model path not configured (set PVEPGENOS_MODEL_PATH)", allow_module_level=True)
    if not os.path.exists(str(env_model_path)):
        pytest.skip(f"Model path not found: {env_model_path}", allow_module_level=True)
    return env_model_path


@pytest.fixture(scope="session")
def require_real_data(env_vcf_path, env_bed_path, env_ref_fasta):
    """真实数据路径不可用时 skip 测试"""
    missing = []
    vals = {
        "PVEPGENOS_VCF_PATH": env_vcf_path,
        "PVEPGENOS_BED_PATH": env_bed_path,
        "PVEPGENOS_REF_FASTA": env_ref_fasta,
    }
    for name, val in vals.items():
        if not val:
            missing.append(name)
        elif not os.path.exists(str(val)):
            missing.append(f"{name} (path not found)")

    if missing:
        pytest.skip(f"Missing data paths: {missing}", allow_module_level=True)

    return dict(
        vcf_path=env_vcf_path,
        bed_path=env_bed_path,
        ref_fasta=env_ref_fasta,
    )


@pytest.fixture(scope="session")
def require_api_service(env_api_url):
    """API 服务不可用时 skip 测试"""
    import httpx
    try:
        resp = httpx.get(f"{env_api_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        pytest.skip(f"API service not available at {env_api_url}: {e}", allow_module_level=True)
