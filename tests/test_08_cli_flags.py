"""
tests/test_08_cli_flags.py
==========================
测试 CLI 参数解析（通用，CPU/GPU 均可）。

覆盖场景
--------
1. --config 必填（缺省时报错）
2. --mode local / api
3. --api-base-url 与 mode=api 联动
4. --seq-builder builtin / genvarloader
5. --devices 单卡 / 多卡
6. --save-haplotypes / --no-save-haplotypes
7. --save-embeddings / --no-save-embeddings
8. 参数组合：api + genvarloader、local + 多卡等
9. resolve_* 辅助函数

前置条件
--------
- 无（仅解析参数，不实际运行 pipeline）

运行（任意节点）
----------------
pytest tests/test_08_cli_flags.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """创建最小有效 config.yaml"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as f:
        f.write("""
vcf_path: /tmp/fake.vcf.gz
bed_path: /tmp/fake.bed
ref_fasta: /tmp/fake.fa
output_dir: /tmp/pVEP_out
model:
  name: TestModel
  path: /tmp/fake_model
  dtype: float32
  batch_size: 8
  mode: local
  device: cpu
  devices: []
window_size: 128
embedding:
  methods: [mean]
  save_interval: 50
seq_builder:
  type: builtin
output:
  save_haplotypes: true
  save_embeddings: true
""")
        path = f.name
    yield path
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ─────────────────────────────────────────────────────────────
# 辅助：调用 parse_args
# ─────────────────────────────────────────────────────────────

def run_parse_args(extra_args=None, config_path=None):
    """模拟命令行调用 parse_args()"""
    from run_pipeline import parse_args

    orig_argv = sys.argv
    try:
        args_list = ["prog"]
        if config_path:
            args_list.extend(["--config", config_path])
        if extra_args:
            args_list.extend(extra_args)
        sys.argv = args_list
        return parse_args()
    finally:
        sys.argv = orig_argv


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────

class TestConfigRequired:
    """--config 必填"""

    def test_missing_config_raises(self):
        """不传 --config 应抛出 SystemExit"""
        orig_argv = sys.argv
        sys.argv = ["prog"]  # 没有 --config
        try:
            with pytest.raises(SystemExit):
                run_parse_args()
        finally:
            sys.argv = orig_argv


class TestModeFlag:
    """--mode 参数"""

    def test_default_mode_none(self, minimal_config):
        """不传 --mode → args.mode = None（由 resolve_mode 读 config）"""
        args = run_parse_args(config_path=minimal_config)
        assert args.mode is None

    def test_explicit_local(self, minimal_config):
        """--mode local"""
        args = run_parse_args(extra_args=["--mode", "local"], config_path=minimal_config)
        assert args.mode == "local"

    def test_explicit_api(self, minimal_config):
        """--mode api"""
        args = run_parse_args(extra_args=["--mode", "api"], config_path=minimal_config)
        assert args.mode == "api"


class TestApiBaseUrl:
    """--api-base-url 参数"""

    def test_api_url_stored(self, minimal_config):
        """--api-base-url 值被正确存储"""
        args = run_parse_args(
            extra_args=["--api-base-url", "http://x:8000"],
            config_path=minimal_config
        )
        assert args.api_base_url == "http://x:8000"


class TestSeqBuilder:
    """--seq-builder 参数"""

    def test_builtin(self, minimal_config):
        """--seq-builder builtin"""
        args = run_parse_args(extra_args=["--seq-builder", "builtin"], config_path=minimal_config)
        assert args.seq_builder == "builtin"

    def test_genvarloader(self, minimal_config):
        """--seq-builder genvarloader"""
        args = run_parse_args(extra_args=["--seq-builder", "genvarloader"], config_path=minimal_config)
        assert args.seq_builder == "genvarloader"

    def test_invalid_builder_raises(self, minimal_config):
        """无效 builder 值应报错"""
        orig_argv = sys.argv
        sys.argv = ["prog", "--config", minimal_config, "--seq-builder", "invalid"]
        try:
            with pytest.raises(SystemExit):
                run_parse_args()
        finally:
            sys.argv = orig_argv


class TestDevicesFlag:
    """--devices 参数"""

    def test_single_device(self, minimal_config):
        """--devices cuda:0"""
        args = run_parse_args(extra_args=["--devices", "cuda:0"], config_path=minimal_config)
        assert args.devices == ["cuda:0"]

    def test_multi_devices(self, minimal_config):
        """--devices cuda:0 cuda:1 cuda:2"""
        args = run_parse_args(
            extra_args=["--devices", "cuda:0", "cuda:1", "cuda:2"],
            config_path=minimal_config
        )
        assert args.devices == ["cuda:0", "cuda:1", "cuda:2"]

    def test_cpu_device(self, minimal_config):
        """--devices cpu"""
        args = run_parse_args(extra_args=["--devices", "cpu"], config_path=minimal_config)
        assert args.devices == ["cpu"]


class TestSaveHaplotypesFlag:
    """--save-haplotypes / --no-save-haplotypes"""

    def test_default_none(self, minimal_config):
        """默认从 config 读取（args 为 None）"""
        args = run_parse_args(config_path=minimal_config)
        assert args.save_haplotypes is None

    def test_no_save_haplotypes(self, minimal_config):
        """--no-save-haplotypes"""
        args = run_parse_args(extra_args=["--no-save-haplotypes"], config_path=minimal_config)
        assert args.save_haplotypes is False

    def test_save_haplotypes(self, minimal_config):
        """--save-haplotypes"""
        args = run_parse_args(extra_args=["--save-haplotypes"], config_path=minimal_config)
        assert args.save_haplotypes is True


class TestSaveEmbeddingsFlag:
    """--save-embeddings / --no-save-embeddings"""

    def test_no_save_embeddings(self, minimal_config):
        """--no-save-embeddings"""
        args = run_parse_args(extra_args=["--no-save-embeddings"], config_path=minimal_config)
        assert args.save_embeddings is False

    def test_save_embeddings(self, minimal_config):
        """--save-embeddings"""
        args = run_parse_args(extra_args=["--save-embeddings"], config_path=minimal_config)
        assert args.save_embeddings is True


class TestFlagCombinations:
    """参数组合"""

    def test_api_mode_full(self, minimal_config):
        """API 模式完整参数"""
        args = run_parse_args(
            extra_args=[
                "--mode", "api",
                "--api-base-url", "http://gpu:8000",
                "--seq-builder", "genvarloader",
                "--no-save-haplotypes",
                "--save-embeddings",
            ],
            config_path=minimal_config
        )
        assert args.mode == "api"
        assert args.api_base_url == "http://gpu:8000"
        assert args.seq_builder == "genvarloader"
        assert args.save_haplotypes is False
        assert args.save_embeddings is True

    def test_local_multi_gpu_full(self, minimal_config):
        """本地多卡完整参数"""
        args = run_parse_args(
            extra_args=[
                "--mode", "local",
                "--devices", "cuda:0", "cuda:1",
                "--seq-builder", "builtin",
                "--save-haplotypes",
                "--no-save-embeddings",
            ],
            config_path=minimal_config
        )
        assert args.mode == "local"
        assert args.devices == ["cuda:0", "cuda:1"]
        assert args.seq_builder == "builtin"
        assert args.save_haplotypes is True
        assert args.save_embeddings is False

    def test_cpu_no_save_anything(self, minimal_config):
        """CPU 模式，不保存任何内容（仅序列构建 debug）"""
        args = run_parse_args(
            extra_args=["--devices", "cpu", "--no-save-haplotypes", "--no-save-embeddings"],
            config_path=minimal_config
        )
        assert args.devices == ["cpu"]
        assert args.save_haplotypes is False
        assert args.save_embeddings is False


class TestResolveFunctions:
    """resolve_* 辅助函数"""

    def test_resolve_mode_local(self, minimal_config):
        from run_pipeline import load_config, resolve_mode
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        assert resolve_mode(args, cfg) == "local"

    def test_resolve_mode_api(self, minimal_config):
        from run_pipeline import load_config, resolve_mode
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config, "--mode", "api", "--api-base-url", "http://x:8000"]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        assert resolve_mode(args, cfg) == "api"

    def test_resolve_api_url_local_mode(self, minimal_config):
        from run_pipeline import load_config, resolve_api_base_url, resolve_mode
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config, "--mode", "local"]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        mode = resolve_mode(args, cfg)
        url = resolve_api_base_url(args, cfg, mode)
        assert url == ""  # local 模式不需要 url

    def test_resolve_flags_defaults(self, minimal_config):
        from run_pipeline import load_config, resolve_flags
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        h, inf, e = resolve_flags(args, cfg)
        assert h is True
        assert inf is True
        assert e is True

    def test_resolve_seq_builder_type(self, minimal_config):
        from run_pipeline import load_config, resolve_seq_builder_type
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        assert resolve_seq_builder_type(args, cfg) == "builtin"

    def test_resolve_devices_from_cli(self, minimal_config):
        from run_pipeline import load_config, resolve_devices
        from run_pipeline import parse_args as _parse

        orig = sys.argv
        sys.argv = ["prog", "--config", minimal_config, "--devices", "cuda:0", "cuda:1"]
        args = _parse()
        cfg = load_config(minimal_config)
        sys.argv = orig

        assert resolve_devices(args, cfg) == ["cuda:0", "cuda:1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
