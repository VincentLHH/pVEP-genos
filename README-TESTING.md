# 测试指南

本目录包含 pVEP-genos 各功能模块的系统化测试脚本，按测试环境分组。

---

## 目录结构

| 文件 | 测试内容 | 运行环境 | 依赖 | 标记 |
|------|----------|----------|------|------|
| `test_01_sequence_builder.py` | builtin 序列构建器 | 任意节点（纯CPU） | mock_fasta | `cpu` |
| `test_02_genvarloader_builder.py` | GenVarLoader 序列构建器 | GPU节点 | genvarloader + 真实数据 | `genvarloader`, `real` |
| `test_03_embedding_manager.py` | 本地 GPU 推理 | GPU节点 | CUDA + 模型权重 | `gpu`, `cpu` |
| `test_04_api_service.py` | API 服务（standalone） | GPU节点 | CUDA + 模型权重（服务进程） | `api`, `gpu` |
| `test_05_api_client.py` | API 客户端 | 任意节点 | httpx（调用远程API） | `api` |
| `test_06_multi_gpu.py` | 多卡推理 | ≥2 GPU节点 | ≥2 CUDA | `gpu`, `multi_gpu`, `genvarloader`, `real` |
| `test_07_sample_checkpoint.py` | 断点续存 | 任意节点 | mock_fasta | `cpu` |
| `test_08_cli_flags.py` | CLI 参数解析 | 任意节点 | pytest | `cpu` |
| `conftest.py` | pytest fixtures、mock数据、环境检测 | — | — | — |

---

## Pytest 标记说明

测试使用统一的 pytest marks 进行环境过滤：

| 标记 | 含义 | 自动 skip 条件 |
|------|------|----------------|
| `gpu` | 需要 CUDA GPU | 无 GPU 时自动跳过 |
| `cpu` | 纯 CPU 测试 | 无条件运行 |
| `genvarloader` | 需要 genvarloader 库 | 库不可用时自动跳过 |
| `multi_gpu` | 需要 ≥2 GPU | 单 GPU 时自动跳过 |
| `api` | 需要 API 服务运行 | 服务不可达时自动跳过 |
| `real` | 需要真实数据文件 | 数据路径未配置时跳过 |
| `slow` | 慢速测试（完整 pipeline） | 可通过 `-m "not slow"` 跳过 |

---

## 快速开始

```bash
cd pVEP-genos

# 运行所有可用测试（自动 skip 无条件的测试）
pytest tests/ -v

# 仅运行 CPU 可执行的测试
pytest tests/ -v -m "cpu"

# 仅运行 GPU 测试
pytest tests/ -v -m "gpu"

# 仅运行 genvarloader 相关测试
pytest tests/ -v -m "genvarloader"

# 仅运行多卡测试
pytest tests/ -v -m "multi_gpu"

# 仅运行 API 相关测试
pytest tests/ -v -m "api"

# 跳过慢速测试
pytest tests/ -v -m "not slow"

# 组合过滤：GPU 但不要多卡
pytest tests/ -v -m "gpu and not multi_gpu"

# 跳过需要 genvarloader 的测试
pytest tests/ -v -m "not genvarloader"
```

---

## GPU 节点测试步骤

### Step 1: 基础环境检测

```bash
# 检查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 检查 genvarloader（可选）
python -c "import genvarloader; print('genvarloader:', genvarloader.__version__)"

# 检查模型路径
ls "$PVEPGENOS_MODEL_PATH"
```

### Step 2: 配置环境变量

```bash
export PVEPGENOS_MODEL_PATH=/path/to/Genos-1.2B
export PVEPGENOS_VCF_PATH=/path/to/your.vcf.gz
export PVEPGENOS_BED_PATH=/path/to/your.bed
export PVEPGENOS_REF_FASTA=/path/to/hg38.fa
```

### Step 3: 按顺序运行测试

```bash
# 1. 序列构建器（纯CPU，无需GPU）
pytest tests/test_01_sequence_builder.py -v

# 2. GenVarLoader（需要 genvarloader 库）
pytest tests/test_02_genvarloader_builder.py -v

# 3. EmbeddingManager 本地推理
pytest tests/test_03_embedding_manager.py -v

# 4. API 服务测试（需要先启动服务，见下方）
pytest tests/test_04_api_service.py -v

# 5. 多卡推理（需要 ≥2 GPU）
pytest tests/test_06_multi_gpu.py -v -s
```

### Step 4: API 服务测试（test_04）

API 服务需要单独启动，然后运行测试：

```bash
# 终端 1：启动服务
python -m api.service \
    --model-name Genos-1.2B \
    --model-path "$PVEPGENOS_MODEL_PATH" \
    --device cuda:0 \
    --port 8000

# 终端 2：运行测试
pytest tests/test_04_api_service.py -v
```

### Step 5: 断点续存测试

```bash
# 使用 mock 数据，无需配置
pytest tests/test_07_sample_checkpoint.py -v

# 如需用真实数据验证：
pytest tests/test_07_sample_checkpoint.py -v \
    --env PVEPGENOS_VCF_PATH=/path/to/vcf \
    --env PVEPGENOS_REF_FASTA=/path/to/ref
```

### Step 6: CLI 参数解析测试

```bash
pytest tests/test_08_cli_flags.py -v
```

---

## 无 GPU 节点测试步骤

### Step 1: API 客户端测试

确保 GPU 节点的 API 服务已启动：

```bash
# 在 GPU 节点已启动：python -m api.service ...
export PVEPGENOS_API_URL=http://gpu-node:8000
pytest tests/test_05_api_client.py -v
```

### Step 2: 纯 CPU 测试

```bash
# 这些测试不需要 GPU
pytest tests/test_01_sequence_builder.py -v
pytest tests/test_07_sample_checkpoint.py -v
pytest tests/test_08_cli_flags.py -v
```

---

## 环境变量说明

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `PVEPGENOS_MODEL_PATH` | 模型权重目录 | 从 config/default.yaml 读取 |
| `PVEPGENOS_VCF_PATH` | VCF 文件路径 | 从 config/default.yaml 读取 |
| `PVEPGENOS_BED_PATH` | BED 文件路径 | 从 config/default.yaml 读取 |
| `PVEPGENOS_REF_FASTA` | 参考基因组 FASTA | 从 config/default.yaml 读取 |
| `PVEPGENOS_API_URL` | API 服务地址 | `http://localhost:8000` |

---

## 常见问题

### Q: test_02 报 "genvarloader not installed"
**A**: genvarloader 需要在 GPU 节点编译安装。如果没有安装，测试会自动 skip：

```bash
# 查看哪些测试会因 genvarloader 跳过
pytest tests/ -v -m "genvarloader" --collect-only

# 跳过所有 genvarloader 相关测试
pytest tests/ -v -m "not genvarloader"
```

### Q: test_03 报 "CUDA not available"
**A**: 当前节点没有 GPU。可以：
1. 使用 API 模式测试（test_05）代替
2. 运行 CPU 版本测试：`pytest tests/test_03_embedding_manager.py -v -k "cpu"`

### Q: test_04 API 服务启动失败
**A**: 确认模型路径正确，CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
ls "$PVEPGENOS_MODEL_PATH"
```

### Q: test_06 多卡测试在单卡 GPU 上运行
**A**: 自动 skip（`skip_if_single_gpu`），不会报错。如需强制测试，请确保有 ≥2 GPU。

### Q: 测试挂起不动
**A**: 可能是模型加载或推理卡住，按 Ctrl+C 终止，检查：
1. 模型路径是否正确
2. GPU 显存是否充足
3. 网络（API 模式）是否连通

### Q: 如何跳过耗时的完整 pipeline 测试？
```bash
pytest tests/ -v -m "not slow"
```

---

## 测试输出解读

```
PASSED  ✅ 功能正常
FAILED  ❌ 功能异常，需要排查
SKIPPED ⏭  条件不满足（如无 GPU），跳过不报错
```

每个测试文件末尾的 `if __name__ == "__main__"` 支持直接运行：

```bash
python tests/test_01_sequence_builder.py -v
```

---

## 架构设计说明

### 统一的 Skip 逻辑（conftest.py）

所有环境检测逻辑集中在 `conftest.py`：

```python
# conftest.py 导出：
HAS_CUDA, GPU_COUNT, HAS_GVL  # 全局常量
skip_if_no_cuda, skip_if_no_gvl, skip_if_single_gpu  # skip helpers
require_model_path, require_real_data, require_api_service  # fixtures
```

各测试文件统一导入使用，避免重复检测：

```python
# 测试文件中：
from tests.conftest import (
    HAS_CUDA, skip_if_no_cuda, skip_if_no_gvl, ...
)

@pytest.mark.gpu
@skip_if_no_cuda
def test_gpu_feature():
    ...
```

### Skip 装饰器优先级

装饰器从下往上执行（最近的对最先检测）：

```python
@pytest.mark.gpu              # 3. 标记为 gpu（可被 -m 过滤）
@skip_if_no_cuda             # 2. 检测 CUDA，不可用则 skip
@pytest.mark.real             # 1. 标记为 real（可被 -m 过滤）
def test_that_needs_gpu_and_real_data():
    ...
```

---

## 贡献新测试

新增测试文件时请：

1. 命名遵循 `test_XX_description.py` 格式（XX = 序号）
2. 使用 `conftest.py` 提供的 fixtures 和 skip helpers
3. GPU 相关测试加 `@pytest.mark.gpu` + `@skip_if_no_cuda`
4. GenVarLoader 相关测试加 `@pytest.mark.genvarloader` + `@skip_if_no_gvl`
5. 多卡测试加 `@pytest.mark.multi_gpu` + `@skip_if_single_gpu`
6. 提供 `if __name__ == "__main__"` 直接运行入口
7. 在本 README 更新测试矩阵表
