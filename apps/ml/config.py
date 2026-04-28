"""
apps.ml.config
==============
ML模块的配置管理。

支持配置文件 + 命令行覆盖。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class DataConfig:
    """数据源配置"""
    # Embedding目录（pipeline输出的 {sample_id}.json 所在目录）
    emb_dir: str = ""
    # 代谢组表格（CSV格式，sample_id列 + 代谢物列）
    metab_file: str = ""
    # 表型组表格（CSV格式，sample_id列 + 表型列）
    pheno_file: str = ""
    # 分类标签文件（CSV格式，sample_id列 + label列）
    label_file: str = ""
    # 样本ID列名
    sample_id_col: str = "sample_id"
    # 标签列名
    label_col: str = "label"

    # -- 变异评分与选择 --
    # 评分策略：
    #   relative : 背景校正 (cos_sim(diff_hap, diff_ref))
    #   absolute : 绝对效应 (||diff_hap1|| + ||diff_hap2||)
    #   weighted : relative + absolute 加权
    #   cascade  : 先用 absolute 粗筛 λk，再用 relative 精选 k
    variant_scoring: str = "absolute"
    # 每个样本选取的 top-k 变异数
    top_k: int = 0  # 0 = 不选，全量聚合
    # 策略 weighted 的权重：score = w * norm(relative) + (1-w) * norm(absolute)
    score_weight: float = 0.5
    # 策略 cascade 的粗筛倍数 λ
    cascade_lambda: float = 2.0
    # top-k 变异 embedding 聚合方式：mean / max
    emb_aggregation: str = "mean"

    # 单个变异的 embedding 表示策略：
    #   mut_hap1:             仅 Mut_hap1
    #   mut_hap2:             仅 Mut_hap2
    #   mut_hap1_hap2_mean:   (Mut_hap1 + Mut_hap2) / 2
    #   mut_hap1_hap2_concat: concat(Mut_hap1, Mut_hap2)  注意：维度翻倍
    #   mut_mean_minus_wt_ref: (Mut_hap1+Mut_hap2)/2 - WT_ref
    #   mut_concat_minus_wt_ref: concat(Mut_hap1-WT_ref, Mut_hap2-WT_ref)  注意：维度翻倍
    emb_representation: str = "mut_hap1"
    # 是否标准化特征
    standardize: bool = True
    # 缺失值填充策略（已废弃，统一使用PreprocessConfig）
    fill_na_strategy: str = "median"  # deprecated, use preprocess instead


@dataclass
class PreprocessConfig:
    """
    预处理配置。

    设计原则：
    1. 基因组embedding：降维（而非填补），只使用训练集统计量防止泄露
    2. 表格数据（代谢组/表型组）：填补（而非降维），只使用训练集统计量防止泄露
    """
    # ========== Embedding 降维配置 ==========
    # 降维策略：pca / none
    emb_reducer: str = "pca"
    # PCA目标维度："auto" -> min(n_samples, n_features)
    # 或指定整数如 128, 256
    emb_n_components: str = "auto"
    # 降维前是否先标准化
    emb_standardize_first: bool = True

    # ========== 表格数据填补配置 ==========
    # 填补策略：median / mean / most_frequent / zero
    tab_impute_strategy: str = "median"

    # ========== 全局 ==========
    # 是否启用预处理
    enabled: bool = True


@dataclass
class CVConfig:
    """交叉验证配置"""
    n_folds: int = 5
    stratified: bool = True  # 按标签分层
    shuffle: bool = True
    random_state: int = 42


@dataclass
class HyperparamConfig:
    """超参搜索空间"""
    svm: Dict = field(default_factory=lambda: {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    })
    logistic_regression: Dict = field(default_factory=lambda: {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0, 1],
        "solver": ["liblinear"],
        "max_iter": [1000],
    })
    xgboost: Dict = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    })
    mlp: Dict = field(default_factory=lambda: {
        "hidden_layer_sizes": [[8], [16], [32], [16, 8]],
        "activation": ["relu", "tanh"],
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "max_iter": [2000],
        "early_stopping": [True],
        "validation_fraction": [0.2],
        "n_iter_no_change": [20],
    })


@dataclass
class AblationConfig:
    """消融实验配置"""
    # 启用哪些模态
    modules: List[str] = field(default_factory=lambda: [
        "genome_only",
        "genome+metab",
        "genome+pheno",
        "all",
    ])
    # 保存消融实验结果
    save_dir: str = "outputs/ablation"


@dataclass
class OutputConfig:
    """输出配置"""
    save_dir: str = "outputs/ml"
    save_best_params: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    verbose: int = 1


@dataclass
class GlobalConfig:
    """全局配置"""
    random_state: int = 42
    n_jobs: int = -1  # 并行任务数，-1使用所有CPU
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    hyperparam: HyperparamConfig = field(default_factory=HyperparamConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> GlobalConfig:
    """从YAML文件加载配置"""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = GlobalConfig()
    for section in ["data", "preprocess", "cv", "hyperparam", "ablation", "output"]:
        if section in raw:
            section_cfg = getattr(cfg, section)
            for k, v in raw[section].items():
                if hasattr(section_cfg, k):
                    setattr(section_cfg, k, v)
                else:
                    # 尝试直接赋值（支持任意字段）
                    setattr(section_cfg, k, v)

    # 全局字段
    for field in ["random_state", "n_jobs"]:
        if field in raw:
            setattr(cfg, field, raw[field])

    return cfg


def save_config(cfg: GlobalConfig, path: str):
    """保存配置到YAML文件"""
    def dataclass_to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    with open(path, "w") as f:
        yaml.dump(dataclass_to_dict(cfg.__dict__), f, default_flow_style=False)
