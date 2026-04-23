# pVEP-genos

基于 Genos-1.2B 模型的全基因组变异 embedding 推理流水线。

给定 VCF + 参考基因组 + BED 区间，自动重建每个样本在每个区间的单倍型序列，通过 Genos-1.2B 推理得到 embedding，向量用于下游 GWAS/功能基因组分析。

---

## 目录

- [架构概览](#架构概览)
- [前置条件](#前置条件)
- [快速开始](#快速开始)
- [完整部署](#完整部署)
- [API 服务](#api-服务)
- [配置说明](#配置说明)
- [输出格式](#输出格式)
- [测试](#测试)
- [交叉验证](#交叉验证)
- [多卡推理](#多卡推理)
- [序列构建器](#序列构建器)

---

## 架构概览

```
VCF ──┐
      │                                    ┌─ GPU 推理 ──┐
BED ──┼──▶ SequenceBuilder ──▶ hap_seq ──▶│ EmbeddingManager │──▶ embeddings
      │    (builtin / genvarloader)       └──────────────┘
Ref ──┘
```

**数据流：**

1. **BED** 指定待分析区间（如 GWAS 显著区域）
2. **VCF** 包含样本的基因型（GT field）和变异信息
3. **SequenceBuilder** 根据基因型重建区间内的单倍型序列：
   - `builtin`：纯 Python 实现，直接从 FASTA + VCF 读取
   - `genvarloader`：GenVarLoader C 扩展加速，适合超大规模 VCF
4. **EmbeddingManager** 调用 Genos-1.2B 推理，输出 mean/max/last_token pooling 的 embedding 向量

---

## 前置条件

### 软件环境

| 依赖 | 说明 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | CUDA 支持（GPU 节点）或 CPU（仅 API 模式） |
| pysam | VCF/FASTA 读写 |
| transformers | Genos 模型加载 |
| tqdm | 进度条 |
| httpx | API 客户端模式 |
| uvicorn | API 服务 |
| fastapi | API 服务框架 |
| pytest | 测试（可选） |

安装：
```bash
pip install -r requirements.txt
```

### 数据文件

| 文件 | 格式 | 说明 |
|------|------|------|
| 参考基因组 | FASTA + .fai | hg38 或 hg19 |
| VCF | VCF.gz + .tbi | 必须包含 GT field |
| BED | BED (tab分隔, 0-based) | 待分析区间 |

---

## 快速开始

### 方式一：GPU 节点（本地推理）

```bash
# 克隆
git clone https://github.com/VincentLHH/pVEP-genos.git
cd pVEP-genos
pip install -r requirements.txt

# 配置（编辑 config/default.yaml）
vim config/default.yaml

# 运行
python run_pipeline.py --config config/default.yaml
```

### 方式二：CPU 节点（调用远程 API）

CPU 节点无需 GPU，通过 HTTP 调用 GPU 节点上的 API 服务：

```bash
# 1. GPU 节点启动 API 服务（见下文"API 服务"章节）

# 2. CPU 节点配置
export PVEPGENOS_API_URL=http://<GPU节点IP>:8000

# 3. 编辑 config，mode 改为 api，填入 api_base_url
#    或通过命令行覆盖：
python run_pipeline.py --config config/default.yaml \
    --mode api \
    --api-base-url http://27.18.114.42:8000
```

> **注意**：GPU 节点的防火墙/安全组需放行对应端口（默认 8000）。

---

## 完整部署

### 1. 准备配置文件

复制并修改 `config/default.yaml`：

```yaml
# 数据文件路径
vcf_path: /path/to/your/chrN.vcf.gz
bed_path: /path/to/your/regions.bed
ref_fasta: /path/to/hg38.fa

# 输出目录
output_dir: outputs

# 序列构建器（builtin 或 genvarloader）
seq_builder:
  type: builtin          # builtin: 纯 Python（通用）
                         # genvarloader: C 扩展（更快，需安装 genvarloader 库）
  gvl_cache_dir: ""      # .gvl 缓存目录，不填默认 /tmp/gvl_cache
  gvl_strandaware: true  # 负链区域是否 reverse complement

# 模型配置
model:
  name: Genos-1.2B
  path: /path/to/Genos-1.2B     # 模型权重目录
  dtype: bfloat16
  batch_size: 32
  mode: local                  # local: 本机推理；api: 调用远程服务
  api_base_url: ""             # mode=api 时填写
  devices: []                  # 多卡列表，如 [cuda:0, cuda:1]
  device: cuda                 # 单卡默认设备

# 窗口大小（单位 bp）
window_size: 128

# Embedding 方法
embedding:
  methods: ["mean"]            # 支持 mean / max / last_token
  save_interval: 50            # 每处理 N 个样本写入一次磁盘

# 输出控制
output:
  save_haplotypes: true        # 是否保存重建后的单倍型序列
  save_embeddings: true        # 是否保存 embedding
```

### 2. 准备参考基因组索引

```bash
# pysam 需要 .fai 索引
samtools faidx /path/to/hg38.fa
```

### 3. 准备 BED 文件

BED 文件为 0-based、半开区间（符合 UCSC 标准）：

```
chr1    100000  100200
chr2    500000  500500
chrX    2000000 2000300
```

### 4. 运行

```bash
python run_pipeline.py --config config/default.yaml
```

输出结构：
```
outputs/
├── sample1/
│   ├── embeddings.json      # embedding 结果
│   └── haplotypes.json      # 重建的单倍型序列（save_haplotypes=true 时）
├── sample2/
│   └── ...
```

---

## API 服务

GPU 节点可启动独立 API 服务，CPU 节点通过 HTTP 调用。

### 启动服务

```bash
python -m api.service \
    --model-name Genos-1.2B \
    --model-path /path/to/Genos-1.2B \
    --device cuda:0 \
    --port 8000
```

> **注意**：服务固定单 worker 运行（不支持 `--workers` 参数），因为 EmbeddingManager 加载在主进程。

### 端点说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 + cache 统计 |
| POST | `/embed` | 批量推理 |
| GET | `/cache/size` | cache 条目数 |
| DELETE | `/cache` | 清空 cache |

### 调用示例

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "seq_dict": {
      "var1||hap1": "ACGTACGT...",
      "var1||hap2": "ACGTTCGT..."
    },
    "methods": ["mean"]
  }'
```

---

## 配置说明

### 环境变量（可选）

| 变量 | 覆盖范围 |
|------|----------|
| `PVEPGENOS_MODEL_PATH` | 模型权重路径 |
| `PVEPGENOS_REF_FASTA` | 参考基因组路径 |
| `PVEPGENOS_VCF_PATH` | VCF 路径 |
| `PVEPGENOS_BED_PATH` | BED 路径 |
| `PVEPGENOS_API_URL` | API 服务地址（CPU 节点用）|

环境变量优先级高于 `config/default.yaml`。

### 命令行覆盖

所有 config 中的参数都可通过命令行覆盖（覆盖优先级：CLI > 环境变量 > 配置文件）：

```bash
python run_pipeline.py --config config/default.yaml \
    --mode api \
    --api-base-url http://192.168.1.100:8000 \
    --devices cuda:0 cuda:1 \
    --no-save-haplotypes
```

---

## 输出格式

### embeddings.json

```json
{
  "metadata": {
    "model": "Genos-1.2B",
    "window_size": 128,
    "methods": ["mean", "max"],
    "dtype": "float32",
    "created_at": "2026-04-23T10:00:00"
  },
  "results": [
    {
      "sample": "sample1",
      "region": "chr1:100000-100200",
      "variants": [
        {
          "var_id": "chr1_101_T_C",
          "pos": 101,
          "ref": "T",
          "alt": "C",
          "gt": "1|0",
          "hap1_mut_seq": "...",
          "hap2_mut_seq": "...",
          "embeddings": {
            "mean": [0.123, -0.456, ...],
            "max": [0.789, 0.012, ...]
          }
        }
      ]
    }
  ]
}
```

### haplotypes.json（可选）

```json
{
  "sample": "sample1",
  "region": "chr1:100000-100200",
  "variants": [
    {
      "var_id": "chr1_101_T_C",
      "gt": "1|0",
      "hap1_mut_seq": "ACGTTGCAT...",   // hap1 的重建序列（ALT）
      "hap2_mut_seq": "ACGTTGCAT..."    // hap2 的重建序列（REF）
    }
  ]
}
```

---

## 测试

```bash
# 运行全部测试（GPU 节点）
pytest tests/ -v

# 仅核心逻辑（CPU 节点，跳过 GPU/API 测试）
pytest tests/test_01_sequence_builder.py -v

# API 服务测试（需先启动服务）
pytest tests/test_04_api_service.py -v

# API 客户端测试（需配置 PVEPGENOS_API_URL）
pytest tests/test_05_api_client.py -v

# GPU 多卡测试（需 ≥2 GPU）
pytest tests/test_06_multi_gpu.py -v

# 交叉验证（CPU 节点，有 genvarloader）：比对 builtin vs genvarloader 序列重建结果
pytest tests/test_09_cross_validation.py -v

# 仅 mock 数据（排除真实数据抽样）
pytest tests/test_09_cross_validation.py -v -k "not real"
```

> **跳过说明**：部分测试依赖 genvarloader 库或 GPU 节点环境，未安装时自动 skip（非失败）。
>
> **交叉验证测试**（`test_09`）用于验证 builtin 和 genvarloader 两种序列构建器对同一输入产生完全一致的输出。在 CPU 节点（有 genvarloader）上运行，覆盖 SNP（4 种基因型）、INDEL（DEL/INS）、多 Variant 叠加等场景。

---

## 交叉验证

`tests/test_09_cross_validation.py` 对比 **builtin `SequenceBuilder`** 和 **genvarloader `GenVarLoaderSequenceBuilder`** 的序列重建输出是否一致，无需人工判断正确性。

**在 CPU 节点（有 genvarloader）上运行**，自动生成配套的 mock VCF + FASTA + BED，对齐两个 builder 的输入，保证比对公平。

**测试场景**：

| 场景 | 说明 |
|------|------|
| SNP homo ALT | 纯合变异，两条单倍型均为 ALT |
| SNP hetero (1\|0) | 杂合，hap1=ALT / hap2=REF |
| SNP hetero (0\|1) | 杂合，hap1=REF / hap2=ALT |
| INDEL DEL/INS | 3bp 缺失、3bp 插入、6bp 缺失 |
| Multi-Variant | 窗口内 2 个 SNP（同/异单倍型叠加） |
| 真实数据 | 随机抽样真实 VCF/BED/FASTA 的 region，验证端到端一致性 |

**比对字段**：`hap1.mut_seq`、`hap2.mut_seq`、`ref_seq`

---

## 多卡推理

```bash
# 使用 2 张 GPU，并行处理不同样本
python run_pipeline.py --config config/default.yaml \
    --devices cuda:0 cuda:1

# 使用 4 张 GPU
python run_pipeline.py --config config/default.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

**并行策略**：样本按 round-robin 分配到各 GPU，全局 embedding cache 通过 `Manager.dict()` 跨进程共享，相同序列不重复推理。

---

## 序列构建器

### builtin（默认）

纯 Python 实现，直接从 FASTA 读取参考序列，根据 VCF 中的 GT 将 ALT/REF 等位序列写入 hap_seq。

**特点**：无外部依赖，通用性强，适合大多数场景。

**逻辑说明**：
- 以 BED 区间为中心，取 ±window_size/2 bp 作为窗口序列
- variant 永远落在窗口正中央（index = window_size/2）
- GT=1|0：hap1 写入 ALT，hap2 写入 REF
- GT=0|1：hap1 写入 REF，hap2 写入 ALT
- GT=1|1：两条 hap 均写入 ALT（纯合 ALT）
- GT=0|0：两条 hap 均写入 REF（纯合 REF）
- 负链区域自动做 reverse complement

### genvarloader

调用 GenVarLoader C 扩展加速 VCF 解析和序列重建，内存效率更高，适合百万级变异的大规模 VCF。

```bash
# 安装 genvarloader
pip install genvarloader

# 配置使用
seq_builder:
  type: genvarloader
  gvl_cache_dir: /path/to/cache    # .gvl 文件缓存目录
  gvl_strandaware: true
  gvl_max_mem: 4g                   # gvl.write 内存上限
```

---

## 常见问题

### Q: VCF 中有多个样本，如何指定处理哪些？

修改 config 中的 `samples` 字段（留空则处理所有样本）。

### Q: embedding 向量维度是多少？

由 Genos-1.2B 模型决定，约为 4096 维（`hidden_size`）。

### Q: cache 是什么？有什么用？

embedding cache 以 `(sequence, method)` 为 key 缓存推理结果，相同序列不重复推理。多卡并行时通过共享内存实现跨进程 cache 复用，显著加速含大量重叠区间的分析。

### Q: 如何清空 cache？

```bash
curl -X DELETE http://localhost:8000/cache
```

### Q: API 服务启动后返回 503 "Model not loaded yet"？

服务可能处于启动过程中。等待看到 `✅ Ready on 0.0.0.0:8000` 后再访问。

### Q: 外部节点无法访问 API 服务？

检查 GPU 节点防火墙/安全组是否放行了服务端口（如 `sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT`）。
