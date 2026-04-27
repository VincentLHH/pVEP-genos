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
      │                   BedSplit          6 种序列
BED ──┼──▶ VariantBedSplit ──────▶ SequenceBuilder ──────▶ EmbeddingExtractor ──▶ embeddings
      │    (upstream / downstream)  (builtin / genvarloader)  (tail pooling + concat)
Ref ──┘
```

**数据流：**

1. **BED** 指定待分析区间（第四列为 `chr_pos_ref_alt`）
2. **BedSplit** 将每个变异拆为 upstream / downstream 两行，精确定位变异区域
3. **VCF** 包含样本的基因型（GT field）和变异信息
4. **SequenceBuilder** 构建 6 种序列（Mut/WT × hap1/hap2/ref）：
   - `builtin`：纯 Python 实现，直接从 FASTA + VCF 读取
   - `genvarloader`：GenVarLoader C 扩展加速，适合超大规模 VCF
5. **EmbeddingExtractor** 对每条序列提取变异区域 embedding：
   - upstream 正链推理，downstream 反向互补链推理
   - 仅取末尾 `w = var_len + 2` 个 token 做 tail pooling
   - concat(emb_up, emb_down) → 最终维度 = hidden_size × 2
   - 全局哈希缓存，相同序列不重复推理

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

# BED 拆分参数
bed_split:
  mode: auto             # auto: 程序内部拆分；presplit: 外部已拆分
  n: 200                 # 侧翼扩展长度（bp），需 >= window_size//2 + max(len(alt),len(ref)) + 2

# 模型配置
model:
  name: Genos-1.2B
  path: /path/to/Genos-1.2B     # 模型权重目录
  dtype: bfloat16
  batch_size: 32                # GPU 推理批次大小（每次前向传播的序列条数）
  mode: local                  # local: 本机推理；api: 调用远程服务
  api_base_url: ""             # mode=api 时填写
  devices: []                  # 多卡列表，如 [cuda:0, cuda:1]
  device: cuda                 # 单卡默认设备

# 每个方向送入模型的序列长度（bp）
# 需与 n 配合：n >= window_size // 2 + max(len(alt), len(ref)) + 2
window_size: 128

# Embedding 提取配置
embedding:
  pooling: mean              # Tail pooling 方法：mean 或 max
  save_interval: 50          # 每处理 N 个 variant 写入一次磁盘
  save_haplotypes: true      # 是否保存重建后的单倍型序列
  do_inference: true         # 是否执行模型推理（false 时仅构建序列）
  save_embeddings: true      # 是否将 embedding 保存到磁盘（do_inference=false 时无效）
  filter_hom_ref: true       # 是否过滤 0|0 变异（false 时所有样本变异数一致）
  use_global_cache: true     # 全局缓存开关
  variant_batch_size: 16     # 跨变异批量推理积累大小
```

### 2. 准备参考基因组索引

```bash
# pysam 需要 .fai 索引
samtools faidx /path/to/hg38.fa
```

### 3. 准备 BED 文件

BED 文件为 0-based、半开区间，标准 3 列格式即可：

```
chr1    100000  100200
chr2    500000  500500
chrX    2000000 2000300
```

程序默认以 `auto` 模式运行，BED 仅定义待分析区间（chrom, start, end），变异信息从 VCF 中自动提取，第四列不影响运行。若使用 `presplit` 模式（`bed_split.mode: presplit`），则需外部预先完成 upstream/downstream 拆分，第四列以 `_upstream`/`_downstream` 结尾。

### 4. 运行

```bash
python run_pipeline.py --config config/default.yaml
```

输出结构：
```
outputs/
├── sample1.json              # 每个样本一个 JSON 文件
├── sample2.json
└── ...
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

### 核心参数详解

| 参数 | 位置 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `n` | `bed_split.n` | int | 200 | BED 拆分侧翼扩展长度（bp）。上游行 = n bp上游 + 变异区域 + 1 bp下游；下游行 = 1 bp上游 + 变异区域 + n bp下游。BED 行总长度 = n + len(ref) + 1 bp，即模型实际输入序列长度 |
| `window_size` | 顶层 | int | 128 | 旧版 pipeline 遗留参数，仅用于已废弃的 `build()` 方法（居中窗口模式）。新版 pipeline（`build_six_seqs` + tail pooling）中不参与序列构建或推理控制 |
| `batch_size` | `model.batch_size` | int | 32 | GPU 推理批次大小，即每次前向传播处理的序列条数。增大可提升 GPU 利用率，但需更多显存。与 `variant_batch_size` 独立 |
| `variant_batch_size` | `embedding.variant_batch_size` | int | 16 | 跨变异批量推理的积累大小。每积累 N 个 variant 的序列后统一送入 GPU 推理。配合 `batch_size` 使用：`variant_batch_size` × 每个variant约8条唯一序列 ≈ 总序列数，需 ≥ `batch_size` 才能充分利用 GPU |
| `do_inference` | `embedding.do_inference` | bool | true | 是否执行模型推理。`false` 时仅构建序列，不运行模型、不产生 embedding。适用于调试序列构建或只保存单倍型 |
| `save_embeddings` | `embedding.save_embeddings` | bool | true | 是否将 embedding 结果保存到磁盘。`do_inference=false` 时此参数无效。设为 `false` 可仅在内存中使用 embedding（如直接传入下游训练） |
| `filter_hom_ref` | `embedding.filter_hom_ref` | bool | true | 是否过滤基因型为 0\|0 的变异。`true` 减少计算量但不同样本变异数可能不同；`false` 保留 0\|0 确保所有样本变异数一致 |
| `save_haplotypes` | `embedding.save_haplotypes` | bool | true | 是否在输出中保存重建的单倍型序列 |

#### n 与 window_size 的关系

```
BED 行长度 = n + len(ref) + 1 bp

n 决定了 BED 行多长（即序列构建时提取多少上下文）。
模型直接以 BED 行序列作为输入，tokenizer 自带 truncation 处理超长序列。
window_size 是旧版 pipeline 的遗留参数，仅用于已废弃的 build() 方法，
在新版 pipeline（build_six_seqs + tail pooling）中不参与序列构建。

典型配置：
  n=200 → BED 行长度 = 201~205 bp（SNP ~202 bp, 6bp DEL ~207 bp）
  n=400 → BED 行长度 = 401~407 bp

选择 n 时应确保 BED 行长度在模型最大输入长度以内，且包含充足上下文。
```

#### batch_size 与 variant_batch_size 的关系

```
batch_size（model.batch_size）：
  每次 GPU 前向传播处理的序列条数。控制 GPU 单次推理的并行度。

variant_batch_size（embedding.variant_batch_size）：
  每积累多少个 variant 后统一推理。控制 CPU/GPU 之间的流水线深度。

协作方式：
  1 个 variant → 约 8 条唯一序列（6种序列，部分因基因型等价去重）
  variant_batch_size=16 → 约 128 条序列积累后统一推理
  batch_size=32 → 128 条序列分 4 批送入 GPU

  若 variant_batch_size 过小（如 1），每次仅 ~8 条序列送入 GPU，
  batch_size=32 的大部分槽位空闲，GPU 利用率低。
```

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
# 常用命令行参数
python run_pipeline.py --config config/default.yaml \
    --mode api \
    --api-base-url http://192.168.1.100:8000 \
    --devices cuda:0 cuda:1 \
    --no-save-haplotypes \
    --no-inference \          # 仅构建序列，不推理
    --no-save-embeddings \    # 推理但不保存到磁盘
    --keep-hom-ref            # 保留 0|0 变异
```

| CLI 参数 | 等价配置 | 说明 |
|----------|----------|------|
| `--do-inference` / `--no-inference` | `embedding.do_inference` | 是否执行推理 |
| `--save-embeddings` / `--no-save-embeddings` | `embedding.save_embeddings` | 是否保存 embedding 到磁盘 |
| `--save-haplotypes` / `--no-save-haplotypes` | `embedding.save_haplotypes` | 是否保存单倍型序列 |
| `--filter-hom-ref` / `--keep-hom-ref` | `embedding.filter_hom_ref` | 是否过滤 0\|0 变异 |

---

## 输出格式

每个样本输出一个 `{sample_id}.json` 文件，结构如下：

```json
{
  "sample_id": "sample1",
  "haplotypes": {
    "chr1_55016418_G_A": {
      "upstream": {
        "hap1": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false },
        "hap2": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false },
        "ref": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false }
      },
      "downstream": {
        "hap1": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false },
        "hap2": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false },
        "ref": { "mut": "ACGT...", "wt": "ACGT...", "wt_is_alias": false }
      }
    }
  },
  "embeddings": {
    "chr1_55016418_G_A": {
      "Genos-1.2B": {
        "Mut_hap1": [0.123, -0.456, ...],
        "WT_hap1": [0.789, 0.012, ...],
        "Mut_hap2": [0.234, -0.567, ...],
        "WT_hap2": [0.890, 0.123, ...],
        "Mut_ref": [0.345, -0.678, ...],
        "WT_ref": [0.901, 0.234, ...]
      }
    }
  }
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `sample_id` | 样本标识 |
| `haplotypes` | 重建的单倍型序列（`save_haplotypes=true` 时存在） |
| `haplotypes.{var_id}.upstream/downstream` | 上游/下游方向的序列 |
| `haplotypes.{var_id}.*.hap1/hap2` | 两条单倍型的 Mut/WT 序列 |
| `haplotypes.{var_id}.*.ref` | 参考基因组背景的 Mut/WT 序列 |
| `haplotypes.{var_id}.*.wt_is_alias` | WT 是否与 Mut 等价（基因型为 hom-ref 时为 true） |
| `embeddings` | Embedding 结果（`save_embeddings=true` 时存在） |
| `embeddings.{var_id}.{model_name}.Mut_hap1` ~ `WT_ref` | 6 个 embedding 向量，每个维度为 hidden_size × 2 |

---

## 测试

```bash
# 运行全部测试（GPU 节点）
pytest tests/ -v

# 仅核心逻辑（CPU 节点，跳过 GPU/API 测试）
pytest tests/ -v -m "cpu"

# 新版 embedding 逻辑测试（CPU 节点，无需 GPU）
pytest tests/test_10_new_embedding.py -v

# API 服务测试（需先启动服务）
pytest tests/test_04_api_service.py -v

# API 客户端测试（需配置 PVEPGENOS_API_URL）
pytest tests/test_05_api_client.py -v

# GPU 多卡测试（需 ≥2 GPU）
pytest tests/test_06_multi_gpu.py -v

# 交叉验证（CPU 节点，有 genvarloader）：比对 builtin vs genvarloader 序列重建结果
pytest tests/test_09_cross_validation.py -v
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

## 个性化 VEP Embedding 设计

### 6 种序列定义

对于每一个变异（及对应的样本基因型），构建 6 条序列：

| 简称 | 定义 | 与基因型的关系 |
|------|------|--------------|
| `Mut_hap1` | 真实样本单倍型1背景下，目标位点**真实状态**的序列 | 依赖 GT（若 hap1 携带 alt 则含 alt，否则含 ref） |
| `WT_hap1`  | 真实样本单倍型1背景下，目标位点**强制为 ref** 的序列 | 依赖 GT |
| `Mut_hap2` | 同上，针对单倍型2 | 依赖 GT |
| `WT_hap2`  | 同上，针对单倍型2 | 依赖 GT |
| `Mut_ref`  | 参考基因组背景，目标位点**强制插入 alt** | 跨样本共享 |
| `WT_ref`   | 纯参考基因组序列 | 跨样本共享 |

**基因型等价规则（省略重复构建）：**
- `0|0`：`Mut_hap1 == WT_hap1`，`Mut_hap2 == WT_hap2`
- `1|0`：`Mut_hap1 != WT_hap1`，`Mut_hap2 == WT_hap2`
- `0|1`：`Mut_hap1 == WT_hap1`，`Mut_hap2 != WT_hap2`
- `1|1`：`Mut_hap1 != WT_hap1`，`Mut_hap2 != WT_hap2`

### BED 拆分规则（upstream / downstream）

原始 BED 的每个变异行（第四列为 `chr_pos_ref_alt`，1-based）在内部拆分为两行。

设变异的 1-based 起始位置为 `POS`，终止位置 `END = POS + len(ref) - 1`，侧翼扩展长度为 `n`：

| 行类型 | start（0-based） | end（0-based，右开） | 用于推理 |
|--------|----------|------|---------|
| `_upstream`   | `POS - n - 1` | `END + 1` | **正链**输入 |
| `_downstream` | `POS - 2`     | `END + n` | **反向互补链**输入 |

> 两行等长：长度 = `n + var_len_ref + 1`（以 ref 长度计算上游行，alt 长度计算下游行，实际等长由设计保证）

**优势**：变异区域始终在序列末尾附近，取末尾 `w = var_len + 2` 个 token 即可精确捕获变异信号，不受上下文噪音稀释。

### Embedding 提取

1. upstream 正链 → 推理 → 取末尾 `w` 个 token → pooling → `emb_up` [D]
2. downstream 反向互补 → 推理 → 取末尾 `w` 个 token → pooling → `emb_down` [D]
3. `concat(emb_up, emb_down)` → 最终 embedding [2D]

**全局哈希缓存**：以序列内容哈希 + w + pooling 方法为 key（`hash_w{w}_{pooling}`），相同序列在相同参数下不重复推理。不同 w 或 pooling 方法即使序列相同也会产生独立的缓存条目，避免语义不同的 embedding 误命中。

### 序列构建器

#### builtin（默认）

纯 Python 实现：
- 从 FASTA 直接提取 WT_ref → 注入 alt 得到 Mut_ref
- 应用除目标变异以外的背景变异（pysam + indel shift 追踪）得到背景序列
- 根据基因型决定 Mut_hap / WT_hap 关系

#### genvarloader

完全依赖 GenVarLoader 提供真实单倍型（Mut_hap）：
- genvarloader 处理全部背景 indel 对齐
- 从 BedRow 第四列名称（`chr_pos_ref_alt_upstream/downstream`）反推变异坐标
- 根据 upstream/downstream 行设计，从序列末尾/开头 offset 定位并移除目标变异 → 得到 WT_hap

```bash
# 安装 genvarloader
pip install genvarloader

# 配置使用
seq_builder:
  type: genvarloader
  gvl_cache_dir: /path/to/cache    # .gvl 文件缓存目录
  gvl_strandaware: false            # 方向由 EmbeddingExtractor 手动控制
  gvl_max_mem: 4g
```

---

## 常见问题

### Q: VCF 中有多个样本，如何指定处理哪些？

当前版本处理 VCF 中的所有样本。如需指定子集，建议先用 `bcftools view -s` 过滤 VCF 后再运行 pipeline。

### Q: embedding 向量维度是多少？

新版 pVEP 框架下，每个变异-样本对输出 6 个 embedding 向量，每个向量维度为 `hidden_size × 2`（concat emb_up + emb_down）。Genos-1.2B 的 `hidden_size` 约为 4096，因此每个向量维度约为 **8192**。

### Q: cache 是什么？有什么用？

新版使用基于序列内容哈希（xxhash64/SHA256）+ 参数（w, pooling）的全局 embedding 缓存。缓存键格式为 `{seq_hash}_w{w}_{pooling_method}`，确保：
- 相同序列在相同参数下只推理一次（无论来自哪个样本、哪种 hap 类型）
- 不同 w（不同变异类型）或不同 pooling 方法不会误命中缓存
- 显著减少重复计算，尤其在多样本共享同一变异时

### Q: 如何清空 cache？

```bash
curl -X DELETE http://localhost:8000/cache
```

### Q: API 服务启动后返回 503 "Model not loaded yet"？

服务可能处于启动过程中。等待看到 `✅ Ready on 0.0.0.0:8000` 后再访问。

### Q: 外部节点无法访问 API 服务？

检查 GPU 节点防火墙/安全组是否放行了服务端口（如 `sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT`）。

---

## ML应用模块

基于变异 embedding 和多组学表格数据的分类模型训练、评估与消融分析。

### 核心设计

- **多组学整合**：基因组 embedding（来自 pVEP-genos pipeline）+ 代谢组 + 表型组，统一为特征矩阵
- **样本级聚合**：每个样本的多个变异 embedding 按 mean/max 聚合为固定维度向量
- **防数据泄露**：PCA 降维和缺失值填补仅在每折训练集上拟合，杜绝测试集信息泄露
- **可扩展模型**：sklearn + PyTorch 分类器统一接口，自动超参搜索

### 目录结构

```
apps/
├── configs/
│   └── default_ml.yaml         # 默认配置文件
└── ml/
    ├── __init__.py
    ├── run_ml.py               # 主入口
    ├── config.py               # 配置管理（dataclass 驱动）
    ├── data_loader.py          # 多组学数据加载（embedding + 代谢组 + 表型组）
    ├── preprocessor.py         # 预处理（防泄露 pipeline：PCA + 填补）
    ├── models.py               # 分类器模型 + 工厂（sklearn + PyTorch）
    ├── trainer.py              # 交叉验证 + 超参搜索
    ├── evaluator.py            # 评估指标与可视化
    └── ablator.py              # 消融实验管理
```

### 数据流

```
Emb JSON ({sample_id}.json)  ──▶  按样本聚合 (mean/max)  ──▶  genome 特征
                                                                      │
                                                                      ▼
代谢组 CSV                   ──▶  加载 + 样本对齐  ──▶  特征拼接 (hstack)
                                                                      │
表型组 CSV                   ──▶  加载 + 样本对齐  ──▶               │
                                                                      ▼
标签 CSV                     ──▶  样本对齐  ──▶  预处理 (PCA + 填补)  ──▶  模型训练
                                                                      │
                                                                      ▼
                                                              评估 + 可视化 + 消融
```

### Embedding 聚合方法

流程：**评分 → top-k 筛选 → 聚合**。

#### 1. 变异评分

对每个样本的每个变异，从 pipeline 输出的 6 向量（Mut_hap1, WT_hap1, Mut_hap2, WT_hap2, Mut_ref, WT_ref）中提取差分向量：

```
diff_hap1 = Mut_hap1 - WT_hap1
diff_hap2 = Mut_hap2 - WT_hap2
diff_ref  = Mut_ref  - WT_ref
```

四种评分策略（`data.variant_scoring`）：

| 策略 | 算法 | 特点 |
|------|------|------|
| `relative` | `score = -mean(cos_sim(diff_hap1, diff_ref), cos_sim(diff_hap2, diff_ref))` | 偏离背景越多分越高，背景校正，但可能忽略绝对效应大的变异 |
| `absolute` | `score = ‖diff_hap1‖ + ‖diff_hap2‖` | 衡量变异绝对效应，无背景校正 |
| `weighted` | 两套分各自 min-max 归一化后加权求和：`w·norm(rel) + (1-w)·norm(abs)`，由 `score_weight` 控制权重 | 平衡背景校正与绝对效应 |
| `cascade` | 先 `absolute` 粗筛 top λk（λ=`cascade_lambda`），再 `relative` 精选 top k | 两阶段筛选，先粗后精 |

所有策略最终得分越高越优先。

#### 2. Top-k 选择

按得分降序选取 `data.top_k` 个变异（`top_k=0` 时不筛选，全量保留）。

#### 3. 聚合

对 top-k 变异的 `Mut_hap1` 向量聚合（`data.emb_aggregation`）：

```
mean: emb_s = mean(emb_{v_1}, ..., emb_{v_k})    # 默认
max:  emb_s = max(emb_{v_1}, ..., emb_{v_k})
```

聚合后每个样本得到一个 `hidden_size × 2` 维的特征向量，进入 PCA 降维后作为基因组模态特征。

### 消融实验模态组合

| 组合 | 说明 |
|------|------|
| `genome_only` | 仅基因组 embedding（降维后） |
| `metab_only` | 仅代谢组特征 |
| `pheno_only` | 仅表型组特征 |
| `genome+metab` | 基因组 + 代谢组 |
| `genome+pheno` | 基因组 + 表型组 |
| `metab+pheno` | 代谢组 + 表型组 |
| `all` | 全模态（基因组 + 代谢组 + 表型组） |

### 快速开始

```bash
# 1. 配置（编辑 apps/configs/default_ml.yaml）
vim apps/configs/default_ml.yaml

# 2. 完整运行（超参搜索 + 消融实验）
python apps/ml/run_ml.py --config apps/configs/default_ml.yaml

# 3. 只跑消融实验（使用已有最佳参数）
python apps/ml/run_ml.py --config apps/configs/default_ml.yaml --ablation-only

# 4. 指定模型
python apps/ml/run_ml.py --config apps/configs/default_ml.yaml --models xgboost svm

# 5. 跳过超参搜索（使用默认参数）
python apps/ml/run_ml.py --config apps/configs/default_ml.yaml --no-hyperparam-search
```

### 配置文件示例

```yaml
# apps/configs/default_ml.yaml

# 数据源
data:
  emb_dir: "/path/to/embeddings"       # emb 目录（包含 {sample_id}.json 文件）
  metab_file: "/path/to/metabolomics.csv"
  pheno_file: "/path/to/phenotypes.csv"
  label_file: "/path/to/labels.csv"
  sample_id_col: "sample_id"
  label_col: "label"
  emb_aggregation: "mean"              # top-k 聚合方式：mean / max
  variant_scoring: "absolute"         # relative / absolute / weighted / cascade
  top_k: 10                           # 选取的变异数（0=不筛选）
  score_weight: 0.5                   # weighted 策略的权重
  cascade_lambda: 2.0                 # cascade 策略的粗筛倍数

# 预处理（仅在训练集上拟合，防止数据泄露）
preprocess:
  enabled: true
  emb_reducer: "pca"                   # pca / none
  emb_n_components: "auto"             # auto 或整数（如 128）
  emb_standardize_first: true
  tab_impute_strategy: "median"        # median / mean / most_frequent / zero

# 交叉验证
cv:
  n_folds: 5
  stratified: true
  shuffle: true

# 消融实验
ablation:
  modules:
    - genome_only
    - genome+metab
    - genome+pheno
    - all
  save_dir: "outputs/ablation"

# 输出
output:
  save_dir: "outputs/ml"

# 全局
random_state: 42
n_jobs: -1
```

### 模型支持

| 模型 | 标识符 | 类型 | 说明 |
|------|--------|------|------|
| SVM | `svm` | sklearn | 支持向量机，适合高维小样本 |
| Logistic Regression | `logistic_regression` | sklearn | 逻辑回归，可解释性强 |
| XGBoost | `xgboost` | sklearn | 梯度提升树，性能强 |
| MLP | `mlp` | sklearn | 多层感知机，带 early stopping |

### 输出说明

```
outputs/ml/
├── run_config.yaml                          # 保存的运行配置
├── {module}_{model}_cv_results.json         # 交叉验证结果
├── {module}_{model}_params.json             # 最佳超参
├── {module}_{model}_predictions.json        # 预测结果
├── {module}_{model}_evaluation.json         # 评估指标
└── outputs/ablation/
    ├── ablation_summary.csv                 # 消融实验汇总表
    ├── best_config.json                     # 最佳模态+模型配置
    ├── full_ablation_results.json           # 完整消融结果
    └── ablation_comparison.png              # 消融对比图
```

### 测试

```bash
# 运行全部测试
pytest tests/ -v

# 跳过 GPU 依赖测试
pytest tests/ -v -m "cpu"
```
