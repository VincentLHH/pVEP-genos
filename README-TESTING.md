# 测试指南

本目录包含 pVEP-genos 各功能模块的系统化测试脚本，按测试环境分组。

---

## 目录结构

| 文件 | 测试内容 | 运行环境 | 依赖 |
|------|----------|----------|------|
| `test_01_sequence_builder.py` | builtin 序列构建器 | 任意节点（纯CPU） | mock_fasta |
| `test_02_genvarloader_builder.py` | GenVarLoader 序列构建器 | GPU节点 | genvarloader + 真实数据 |
| `test_03_embedding_manager.py` | 本地 GPU 推理 | GPU节点 | CUDA + 模型权重 |
| `test_04_api_service.py` | API 服务（standalone） | GPU节点 | CUDA + 模型权重（服务进程） |
| `test_05_api_client.py` | API 客户端 | 任意节点 | httpx（调用远程API） |
| `test_06_multi_gpu.py` | 多卡推理 | ≥2 GPU节点 | ≥2 CUDA |
| `test_07_sample_checkpoint.py` | 断点续存 | 任意节点 | mock_fasta |
| `test_08_cli_flags.py` | CLI 参数解析 | 任意节点 | pytest |
| `conftest.py` | pytest fixtures、mock数据、环境检测 | — | — |

---

## 快速开始

```bash
cd pVEP-genos

# 运行所有可用测试（自动 skip 无条件的测试）
pytest tests/ -v

# 仅运行 CPU 可执行的测试
pytest tests/ -v -m "not gpu and not genvarloader"
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
ls /path/to/Genos-1.2B/
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

### Step 2: 序列构建器和 checkpoint 测试

```bash
# 这两个测试不需要 GPU
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
**A**: genvarloader 需要在 GPU 节点编译安装。如果没有安装，跳过该测试：
```bash
pytest tests/test_02_genvarloader_builder.py -v --ignore-glob="*genvarloader*"
```

### Q: test_03 报 "CUDA not available"
**A**: 当前节点没有 GPU。使用 API 模式测试（test_05）代替。

### Q: test_04 API 服务启动失败
**A**: 确认模型路径正确，CUDA 可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
ls "$PVEPGENOS_MODEL_PATH"
```

### Q: test_06 多卡测试在单卡 GPU 上运行
**A**: 自动 skip（`skip_if_single_gpu`），不会报错。如需强制测试，修改 `GPU_COUNT` 检测条件。

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

## 贡献新测试

新增测试文件时请：
1. 命名遵循 `test_XX_description.py` 格式（XX = 序号）
2. 使用 `conftest.py` 提供的 fixtures
3. GPU 相关测试加 `@pytest.mark.skipif(not HAS_CUDA)`
4. 提供 `if __name__ == "__main__"` 直接运行入口
5. 在本 README 更新测试矩阵表
