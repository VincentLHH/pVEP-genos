"""
apps.ml.data_loader
===================
多组学数据加载器。

整合：
1. 基因组：embedding向量（从pipeline输出的embeddings.json加载）
2. 代谢组：表格数据（CSV）
3. 表型组：表格数据（CSV）

输出：标准化的特征矩阵 X 和标签向量 y
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MultiOmicsDataLoader:
    """
    多组学数据加载器。

    用法：
        loader = MultiOmicsDataLoader(config)
        X, y, feature_names, sample_ids = loader.load_all()
    """

    def __init__(self, data_cfg):
        """
        Args:
            data_cfg: apps.ml.config.DataConfig 实例
        """
        self.cfg = data_cfg
        self._emb_features = None
        self._metab_features = None
        self._pheno_features = None
        self._labels = None
        self._sample_ids = None
        self._feature_names = None

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        加载全部数据并整合。

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            feature_names: 特征名列表
            sample_ids: 样本ID列表
        """
        self._load_labels()
        self._load_emb()
        self._load_metab()
        self._load_pheno()

        # 对齐样本顺序
        self._align_samples()

        # 整合特征
        X = self._assemble_features()
        feature_names = self._build_feature_names()

        # 处理缺失值
        X = self._fill_na(X)

        return X, self._labels.values, feature_names, self._sample_ids

    def load_all_with_metadata(self) -> Dict:
        """
        加载全部数据，并返回详细元信息。

        Returns:
            dict: {
                "X": 特征矩阵,
                "y": 标签向量,
                "feature_names": 特征名列表,
                "sample_ids": 样本ID列表,
                "emb_indices": genome特征列索引,
                "metab_indices": 代谢组特征列索引,
                "pheno_indices": 表型组特征列索引,
                "emb_dim": genome原始维度,
            }
        """
        X, y, feature_names, sample_ids = self.load_all()

        # 推断各模态列索引
        emb_indices = [i for i, name in enumerate(feature_names) if name.startswith("emb_")]
        metab_indices = [i for i, name in enumerate(feature_names) if name.startswith("metab_")]
        pheno_indices = [i for i, name in enumerate(feature_names) if name.startswith("pheno_")]

        return {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "sample_ids": sample_ids,
            "emb_indices": emb_indices,
            "metab_indices": metab_indices,
            "pheno_indices": pheno_indices,
            "emb_dim": len(emb_indices) if emb_indices else 0,
        }

    def load_subset(self, modules: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        加载指定模态子集（用于消融实验）。

        Args:
            modules: 模态列表，如 ["genome", "metab", "pheno"]

        Returns:
            同 load_all()
        """
        self.load_all()
        X = self._assemble_features(modules)
        feature_names = self._build_feature_names(modules)
        X = self._fill_na(X)
        return X, self._labels.values, feature_names, self._sample_ids

    def _load_labels(self):
        """加载标签"""
        if not self.cfg.label_file:
            raise ValueError("label_file 未配置")

        df = pd.read_csv(self.cfg.label_file)
        if self.cfg.sample_id_col not in df.columns:
            raise ValueError(f"标签文件中未找到列: {self.cfg.sample_id_col}")
        if self.cfg.label_col not in df.columns:
            raise ValueError(f"标签文件中未找到列: {self.cfg.label_col}")

        self._labels = df.set_index(self.cfg.sample_id_col)[self.cfg.label_col]

    def _load_emb(self):
        """加载基因组embedding"""
        if not self.cfg.emb_dir:
            warnings.warn("emb_dir 未配置，跳过基因组特征")
            self._emb_features = pd.DataFrame(index=self._labels.index)
            return

        emb_dir = Path(self.cfg.emb_dir)
        if not emb_dir.exists():
            raise FileNotFoundError(f"Embedding目录不存在: {emb_dir}")

        emb_dict = {}
        for sample_dir in emb_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            emb_file = sample_dir / "embeddings.json"
            if not emb_file.exists():
                continue

            with open(emb_file) as f:
                data = json.load(f)

            # 聚合所有region的embedding
            all_embs = []
            for result in data.get("results", []):
                for var in result.get("variants", []):
                    emb = var.get("embeddings", {}).get("mean", [])
                    if emb:
                        all_embs.append(emb)

            if not all_embs:
                warnings.warn(f"样本 {sample_dir.name} 无有效embedding")
                continue

            if self.cfg.emb_aggregation == "mean":
                sample_emb = np.mean(all_embs, axis=0)
            elif self.cfg.emb_aggregation == "max":
                sample_emb = np.max(all_embs, axis=0)
            else:
                sample_emb = np.mean(all_embs, axis=0)

            emb_dict[sample_dir.name] = sample_emb

        if not emb_dict:
            raise ValueError("未找到任何有效的embedding数据")

        # 转为DataFrame
        n_dims = len(next(iter(emb_dict.values())))
        emb_df = pd.DataFrame(
            emb_dict,
            index=[f"emb_{i}" for i in range(n_dims)]
        ).T
        emb_df.index.name = self.cfg.sample_id_col

        self._emb_features = emb_df

    def _load_metab(self):
        """加载代谢组数据"""
        if not self.cfg.metab_file:
            warnings.warn("metab_file 未配置，跳过代谢组特征")
            self._metab_features = pd.DataFrame(index=self._labels.index)
            return

        df = pd.read_csv(self.cfg.metab_file)
        if self.cfg.sample_id_col not in df.columns:
            warnings.warn(f"代谢组文件中未找到列: {self.cfg.sample_id_col}")
            self._metab_features = pd.DataFrame(index=self._labels.index)
            return

        # 设置索引，移除ID列
        df = df.set_index(self.cfg.sample_id_col)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._metab_features = df[numeric_cols]

    def _load_pheno(self):
        """加载表型组数据"""
        if not self.cfg.pheno_file:
            warnings.warn("pheno_file 未配置，跳过表型组特征")
            self._pheno_features = pd.DataFrame(index=self._labels.index)
            return

        df = pd.read_csv(self.cfg.pheno_file)
        if self.cfg.sample_id_col not in df.columns:
            warnings.warn(f"表型组文件中未找到列: {self.cfg.sample_id_col}")
            self._pheno_features = pd.DataFrame(index=self._labels.index)
            return

        df = df.set_index(self.cfg.sample_id_col)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._pheno_features = df[numeric_cols]

    def _align_samples(self):
        """按标签对齐所有数据源的样本顺序"""
        common_idx = self._labels.index

        if self._emb_features is not None and not self._emb_features.empty:
            common_idx = common_idx.intersection(self._emb_features.index)
        if self._metab_features is not None and not self._metab_features.empty:
            common_idx = common_idx.intersection(self._metab_features.index)
        if self._pheno_features is not None and not self._pheno_features.empty:
            common_idx = common_idx.intersection(self._pheno_features.index)

        if len(common_idx) == 0:
            raise ValueError("各数据源无共同样本")

        self._sample_ids = list(common_idx)
        self._labels = self._labels.loc[common_idx]

        if self._emb_features is not None:
            self._emb_features = self._emb_features.loc[common_idx]
        if self._metab_features is not None:
            self._metab_features = self._metab_features.loc[common_idx]
        if self._pheno_features is not None:
            self._pheno_features = self._pheno_features.loc[common_idx]

    def _assemble_features(self, modules: Optional[List[str]] = None) -> np.ndarray:
        """
        整合各模态特征。

        Args:
            modules: 启用的模态列表，None表示全部

        Returns:
            特征矩阵
        """
        if modules is None:
            modules = ["genome", "metab", "pheno"]

        parts = []
        if "genome" in modules:
            parts.append(self._emb_features.values)
        if "metab" in modules:
            parts.append(self._metab_features.values)
        if "pheno" in modules:
            parts.append(self._pheno_features.values)

        return np.hstack(parts) if parts else np.array([])

    def _build_feature_names(self, modules: Optional[List[str]] = None) -> List[str]:
        """构建特征名列表"""
        if modules is None:
            modules = ["genome", "metab", "pheno"]

        names = []
        if "genome" in modules and self._emb_features is not None:
            names.extend(self._emb_features.columns.tolist())
        if "metab" in modules and self._metab_features is not None:
            names.extend(self._metab_features.columns.tolist())
        if "pheno" in modules and self._pheno_features is not None:
            names.extend(self._pheno_features.columns.tolist())

        return names

    def _fill_na(self, X: np.ndarray) -> np.ndarray:
        """填充缺失值"""
        strategy = self.cfg.fill_na_strategy
        if strategy == "median":
            fill_val = np.nanmedian(X)
        elif strategy == "mean":
            fill_val = np.nanmean(X)
        elif strategy == "zero":
            fill_val = 0.0
        else:
            fill_val = 0.0

        return np.nan_to_num(X, nan=fill_val)


def load_and_split(
    data_cfg,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    modules: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    加载数据并分割训练/测试集。

    Args:
        data_cfg: DataConfig实例
        test_size: 测试集比例
        stratify: 是否按标签分层
        random_state: 随机种子
        modules: 启用的模态

    Returns:
        X_train, X_test, y_train, y_test, train_ids, test_ids
    """
    from sklearn.model_selection import train_test_split

    loader = MultiOmicsDataLoader(data_cfg)
    X, y, _, sample_ids = loader.load_subset(modules) if modules else loader.load_all()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, sample_ids,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, idx_train, idx_test
