"""
apps.ml.preprocessor
====================
多组学数据预处理模块。

设计原则
--------
1. **基因组embedding**：降维（而非填补）
   - 默认 PCA 到 min(n_samples, n_features) 维度
   - 只使用训练集的 fit 参数，防止数据泄露

2. **表格数据（代谢组/表型组）**：填补（而非降维）
   - 策略可选：中位数 / 平均值 / 众数
   - 只使用训练集统计量，防止数据泄露

使用 Pipeline 封装，确保 fit/transform 顺序正确。
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 自定义 transformers
# ============================================================

class EmbeddingReducer(BaseEstimator, TransformerMixin):
    """
    基因组embedding降维器。

    参数
    ----
    n_components: int or None
        目标维度。默认 "auto" -> min(n_samples, n_features)
    standardize_first: bool
        是否先标准化。默认 True

    注意：只使用训练集的 PCA 参数，确保无数据泄露。
    """

    def __init__(self, n_components="auto", standardize_first=True, random_state=42):
        self.n_components = n_components
        self.standardize_first = standardize_first
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # NaN 填充：embedding 不应有缺失，若出现则填 0（保守选择）
        self.nan_mask_ = np.isnan(X)
        if self.nan_mask_.any():
            X = np.nan_to_num(X, nan=0.0)

        # 确定目标维度
        if self.n_components == "auto":
            self.n_components_ = min(n_samples, n_features)
        else:
            self.n_components_ = min(self.n_components, n_samples, n_features)

        # 标准化
        if self.standardize_first:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_scaled = X

        # PCA
        self.pca_ = PCA(n_components=self.n_components_, random_state=self.random_state)
        self.pca_.fit(X_scaled)

        # 保存统计量
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        return self

    def transform(self, X):
        X = np.asarray(X)
        # 填 NaN 为 0（与 fit 阶段一致）
        X = np.nan_to_num(X, nan=0.0)

        if self.standardize_first and self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        X_reduced = self.pca_.transform(X_scaled)

        return X_reduced

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_explained_variance_ratio(self):
        """返回各主成分解释方差比例"""
        if self.pca_ is None:
            return None
        return self.pca_.explained_variance_ratio_

    def get_total_explained_variance(self):
        """返回累计解释方差"""
        if self.pca_ is None:
            return None
        return float(np.sum(self.pca_.explained_variance_ratio_))


class TableImputer(BaseEstimator, TransformerMixin):
    """
    表格数据填补器（用于代谢组/表型组）。

    参数
    ----
    strategy: str
        填补策略：median / mean / most_frequent

    注意：只使用训练集统计量，确保无数据泄露。
    """

    def __init__(self, strategy="median"):
        valid_strategies = ["median", "mean", "most_frequent", "zero"]
        if strategy not in valid_strategies:
            raise ValueError(f"strategy必须是 {valid_strategies} 之一")
        self.strategy = strategy

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if self.strategy == "median":
            self.fill_value_ = np.nanmedian(X, axis=0)  # per-column
        elif self.strategy == "mean":
            self.fill_value_ = np.nanmean(X, axis=0)     # per-column
        elif self.strategy == "most_frequent":
            # 每列独立众数（忽略NaN）
            n_cols = X.shape[1]
            self.fill_value_ = np.zeros(n_cols)
            for j in range(n_cols):
                col = X[:, j]
                col_valid = col[~np.isnan(col)]
                if len(col_valid) == 0:
                    self.fill_value_[j] = 0.0
                else:
                    from collections import Counter
                    self.fill_value_[j] = Counter(col_valid).most_common(1)[0][0]
        elif self.strategy == "zero":
            self.fill_value_ = 0.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X = np.where(nan_mask, self.fill_value_, X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    列选择器：按索引或布尔掩码选择列。

    用于将混合特征矩阵拆分为 genome / metab / pheno 部分。
    """

    def __init__(self, col_indices=None):
        """
        Args:
            col_indices: list of int or bool mask
        """
        self.col_indices = col_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.col_indices is None:
            return X
        return X[:, self.col_indices]


# ============================================================
# 预处理器封装
# ============================================================

class MultiOmicsPreprocessor:
    """
    多组学数据预处理器。

    自动将特征拆分为：
    - genome: embedding特征 -> 降维
    - tabular: 代谢组/表型组 -> 填补

    用法：
        preprocessor = MultiOmicsPreprocessor(
            emb_n_components="auto",      # PCA目标维度
            tab_strategy="median",        # 表格数据填补策略
        )
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    """

    def __init__(
        self,
        emb_n_components="auto",
        emb_standardize_first=True,
        tab_strategy="median",
        emb_feature_indices=None,
        tab_feature_indices=None,
        random_state=42,
    ):
        """
        Args:
            emb_n_components: genome embedding 降维目标维度
            emb_standardize_first: genome embedding 是否先标准化
            tab_strategy: 表格数据填补策略 (median/mean/most_frequent/zero)
            emb_feature_indices: genome特征列索引 (None则根据feature_names推断)
            tab_feature_indices: 表格特征列索引
            random_state: 随机种子，传递给 PCA 等随机组件
        """
        self.emb_n_components = emb_n_components
        self.emb_standardize_first = emb_standardize_first
        self.tab_strategy = tab_strategy
        self.emb_feature_indices = emb_feature_indices
        self.tab_feature_indices = tab_feature_indices
        self.random_state = random_state

    def fit(self, X, feature_names=None):
        """
        拟合预处理器。

        Args:
            X: 特征矩阵 (n_samples, n_features)
            feature_names: 特征名列表，用于自动推断列索引
        """
        X = np.asarray(X)

        # 自动推断列索引
        if feature_names is not None:
            self._infer_column_indices(feature_names)
        elif self.emb_feature_indices is None and self.tab_feature_indices is None:
            # 无特征名且无手动索引 -> 根据列数启发式分割
            n_features = X.shape[1]
            n_emb = min(X.shape[0], n_features // 2)
            self.emb_feature_indices = list(range(n_emb))
            self.tab_feature_indices = list(range(n_emb, n_features))

        # 构建 pipeline
        self._build_pipeline()

        # 分别 fit 各子 pipeline
        if self.emb_pipeline_ is not None:
            X_emb = X[:, self.emb_feature_indices]
            self.emb_pipeline_.fit(X_emb)

        if self.tab_pipeline_ is not None:
            X_tab = X[:, self.tab_feature_indices]
            self.tab_pipeline_.fit(X_tab)

        return self

    def _infer_column_indices(self, feature_names):
        """根据特征名推断各模态列索引"""
        if self.emb_feature_indices is None:
            self.emb_feature_indices = [
                i for i, name in enumerate(feature_names)
                if name.startswith("emb_")
            ]
        if self.tab_feature_indices is None:
            self.tab_feature_indices = [
                i for i, name in enumerate(feature_names)
                if not name.startswith("emb_")
            ]

    def _build_pipeline(self):
        # genome embedding -> 降维
        if self.emb_feature_indices:
            self.emb_pipeline_ = make_pipeline(
                EmbeddingReducer(
                    n_components=self.emb_n_components,
                    standardize_first=self.emb_standardize_first,
                    random_state=self.random_state,
                ),
            )
        else:
            self.emb_pipeline_ = None

        # tabular data -> 填补
        if self.tab_feature_indices:
            self.tab_pipeline_ = make_pipeline(
                TableImputer(strategy=self.tab_strategy),
            )
        else:
            self.tab_pipeline_ = None

    def fit_transform(self, X, feature_names=None):
        """fit + transform"""
        self.fit(X, feature_names)
        return self.transform(X)

    def transform(self, X):
        """
        转换数据。

        Returns:
            预处理后的特征矩阵
        """
        X = np.asarray(X)
        parts = []

        if self.emb_pipeline_ is not None:
            X_emb = X[:, self.emb_feature_indices]
            emb_part = self.emb_pipeline_.transform(X_emb)
            parts.append(emb_part)

        if self.tab_pipeline_ is not None:
            X_tab = X[:, self.tab_feature_indices]
            tab_part = self.tab_pipeline_.transform(X_tab)
            parts.append(tab_part)

        if not parts:
            return X

        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        """获取输出特征名"""
        names = []

        if self.emb_pipeline_ is not None and self.emb_feature_indices:
            reducer = self.emb_pipeline_.named_steps["embeddingreducer"]
            n_out = reducer.n_components_
            names.extend([f"emb_pca_{i}" for i in range(n_out)])

        if self.tab_pipeline_ is not None and self.tab_feature_indices:
            # 保留原始列名
            if input_features is not None:
                names.extend([input_features[i] for i in self.tab_feature_indices])
            else:
                n_tab = len(self.tab_feature_indices)
                names.extend([f"tab_{i}" for i in range(n_tab)])

        return names

    def get_emb_explained_variance(self):
        """获取embedding降维的解释方差"""
        if self.emb_pipeline_ is None:
            return None
        reducer = self.emb_pipeline_.named_steps["embeddingreducer"]
        return {
            "explained_variance_ratio": reducer.get_explained_variance_ratio(),
            "total_explained_variance": reducer.get_total_explained_variance(),
        }


# ============================================================
# 简化版：直接处理特征矩阵
# ============================================================

def preprocess_X(
    X,
    mode="all",
    emb_n_components="auto",
    tab_strategy="median",
    return_preprocessor=False,
):
    """
    便捷函数：对特征矩阵进行预处理。

    参数
    ----
    X: 特征矩阵 (n_samples, n_features)
    mode: 处理模式
        - "emb_only": 只处理embedding特征（假设全是embedding）
        - "tab_only": 只处理表格特征（假设全是表格数据）
        - "all": 自动检测（emb_开头的为embedding，其余为表格）
    emb_n_components: embedding降维目标维度
    tab_strategy: 表格填补策略
    return_preprocessor: 是否返回预处理器

    Returns
    -------
    X_processed 或 (X_processed, preprocessor)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # 检测列类型
    emb_cols = []
    tab_cols = []

    # 假设特征名格式为 emb_xxx / metab_xxx / pheno_xxx
    # 如果没有特征名，根据位置分割（前半部分为emb，后半部分为tab）
    for i in range(n_features):
        # 简单启发式：前n_samples列视为emb（高维），其余视为tab
        if i < min(n_samples, n_features // 2):
            emb_cols.append(i)
        else:
            tab_cols.append(i)

    parts = []
    preprocessor_info = {}

    # Embedding降维
    if emb_cols and mode in ["emb_only", "all"]:
        X_emb = X[:, emb_cols]
        n_comp = min(emb_n_components, n_samples, len(emb_cols)) if emb_n_components != "auto" else min(n_samples, len(emb_cols))
        reducer = EmbeddingReducer(n_components=n_comp)
        X_emb_reduced = reducer.fit_transform(X_emb)
        parts.append(X_emb_reduced)
        preprocessor_info["emb_reducer"] = reducer

    # 表格填补
    if tab_cols and mode in ["tab_only", "all"]:
        X_tab = X[:, tab_cols]
        imputer = TableImputer(strategy=tab_strategy)
        X_tab_imputed = imputer.fit_transform(X_tab)
        parts.append(X_tab_imputed)
        preprocessor_info["tab_imputer"] = imputer

    if not parts:
        return X if not return_preprocessor else (X, None)

    X_processed = np.hstack(parts)

    if return_preprocessor:
        return X_processed, preprocessor_info
    return X_processed


def preprocess_cv(
    X_train,
    y_train,
    X_test=None,
    mode="all",
    emb_n_components="auto",
    tab_strategy="median",
):
    """
    交叉验证友好的预处理。

    确保：
    1. 预处理器只在训练集上fit
    2. 测试集用训练集的统计量transform

    Args:
        X_train: 训练集特征
        y_train: 训练集标签（用于分层分割）
        X_test: 测试集特征（可选）
        mode: 同 preprocess_X
        emb_n_components: 同上
        tab_strategy: 同上

    Returns:
        (X_train_processed, X_test_processed) 或 (X_train_processed,)
    """
    preprocessor = MultiOmicsPreprocessor(
        emb_n_components=emb_n_components,
        tab_strategy=tab_strategy,
    )

    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)

    if X_test is not None:
        X_test_proc = preprocessor.transform(X_test)
        return X_train_proc, X_test_proc

    return X_train_proc
