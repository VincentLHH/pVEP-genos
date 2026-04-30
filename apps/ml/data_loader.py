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

    加载流程：先探查各数据源可用样本ID → 求交集 → 只加载公共样本数据。

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
        self._loaded = False

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        加载全部数据并整合。

        流程：先对齐样本ID，再按需加载，避免浪费内存。
        首次调用后缓存全部数据，后续调用直接返回缓存。

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            feature_names: 特征名列表
            sample_ids: 样本ID列表
        """
        if self._loaded:
            X = self._assemble_features()
            return X, self._labels.values, self._build_feature_names(), self._sample_ids

        # 第一步：加载标签并探查各源可用ID
        self._load_labels()
        common_ids = self._peek_and_align_ids()

        # 第二步：只加载公共样本的数据
        self._load_emb(common_ids)
        self._load_metab(common_ids)
        self._load_pheno(common_ids)

        # 整合特征
        self._loaded = True
        X = self._assemble_features()
        feature_names = self._build_feature_names()

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
        return X, self._labels.values, feature_names, self._sample_ids

    # ------------------------------------------------------------------
    # 标签加载
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # ID 探查与对齐
    # ------------------------------------------------------------------

    def _peek_and_align_ids(self) -> pd.Index:
        """
        先探查各数据源可用样本ID，求交集，再裁剪标签。
        返回公共样本ID（已排序），后续加载只处理这些样本。
        """
        common_idx = self._labels.index

        # 探查 embedding 可用ID（只列文件名，不解析内容）
        if self.cfg.emb_dir:
            emb_dir = Path(self.cfg.emb_dir)
            if emb_dir.exists():
                emb_ids = pd.Index(
                    f.stem for f in emb_dir.glob("*.json")
                )
                common_idx = common_idx.intersection(emb_ids)

        # 探查代谢组可用ID（只读 sample_id 列）
        if self.cfg.metab_file:
            metab_path = Path(self.cfg.metab_file)
            if metab_path.exists():
                id_col = self.cfg.sample_id_col
                avail_ids = pd.read_csv(metab_path, usecols=[id_col])[id_col]
                common_idx = common_idx.intersection(pd.Index(avail_ids))

        # 探查表型组可用ID（只读 sample_id 列）
        if self.cfg.pheno_file:
            pheno_path = Path(self.cfg.pheno_file)
            if pheno_path.exists():
                id_col = self.cfg.sample_id_col
                avail_ids = pd.read_csv(pheno_path, usecols=[id_col])[id_col]
                common_idx = common_idx.intersection(pd.Index(avail_ids))

        if len(common_idx) == 0:
            raise ValueError("各数据源无共同样本")

        common_idx = common_idx.sort_values()
        self._sample_ids = list(common_idx)
        self._labels = self._labels.loc[common_idx]
        return common_idx

    # ------------------------------------------------------------------
    # 各模态加载（只加载 common_ids 内的样本）
    # ------------------------------------------------------------------

    def _load_emb(self, common_ids: pd.Index):
        if not self.cfg.emb_dir:
            warnings.warn("emb_dir 未配置，跳过基因组特征")
            self._emb_features = pd.DataFrame(index=common_ids)
            return

        emb_dir = Path(self.cfg.emb_dir)
        if not emb_dir.exists():
            raise FileNotFoundError(f"Embedding目录不存在: {emb_dir}")

        id_set = set(common_ids)
        emb_dict = {}
        for emb_file in sorted(emb_dir.glob("*.json")):
            sample_id = emb_file.stem
            if sample_id not in id_set:
                continue

            with open(emb_file) as f:
                data = json.load(f)

            embeddings_data = data.get("embeddings", {})
            rep_strategy = getattr(self.cfg, "emb_representation", "mut_hap1")
            variant_data: list = []  # [(emb_dict, rep_vec), ...]
            for var_id, models_dict in embeddings_data.items():
                if not models_dict:
                    continue
                emb = next(iter(models_dict.values()))
                rep_vec = self._build_rep_vector(emb, rep_strategy)
                if rep_vec is None:
                    continue
                variant_data.append((emb, rep_vec))

            if not variant_data:
                warnings.warn(f"样本 {sample_id} 无有效embedding")
                continue

            top_embs = self._select_top_variants(variant_data)
            if self.cfg.emb_aggregation == "max":
                sample_emb = np.max(top_embs, axis=0)
            else:
                sample_emb = np.mean(top_embs, axis=0)

            emb_dict[sample_id] = sample_emb

        if not emb_dict:
            raise ValueError("未找到任何有效的embedding数据")

        n_dims = len(next(iter(emb_dict.values())))
        emb_df = pd.DataFrame(
            emb_dict,
            index=[f"emb_{i}" for i in range(n_dims)]
        ).T
        emb_df.index.name = self.cfg.sample_id_col

        self._emb_features = emb_df.loc[emb_df.index.isin(common_ids)]
        self._emb_features = self._emb_features.loc[common_ids]

    def _load_metab(self, common_ids: pd.Index):
        """
        加载代谢组数据，只保留 common_ids 中的样本。

        Args:
            common_ids: 公共样本ID索引
        """
        if not self.cfg.metab_file:
            warnings.warn("metab_file 未配置，跳过代谢组特征")
            self._metab_features = pd.DataFrame(index=common_ids)
            return

        df = pd.read_csv(self.cfg.metab_file)
        if self.cfg.sample_id_col not in df.columns:
            warnings.warn(f"代谢组文件中未找到列: {self.cfg.sample_id_col}")
            self._metab_features = pd.DataFrame(index=common_ids)
            return

        # 设置索引，移除ID列，只保留公共样本
        df = df.set_index(self.cfg.sample_id_col)
        df = df.loc[df.index.isin(common_ids)]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._metab_features = df[numeric_cols].loc[common_ids]

    def _load_pheno(self, common_ids: pd.Index):
        """
        加载表型组数据，只保留 common_ids 中的样本。

        Args:
            common_ids: 公共样本ID索引
        """
        if not self.cfg.pheno_file:
            warnings.warn("pheno_file 未配置，跳过表型组特征")
            self._pheno_features = pd.DataFrame(index=common_ids)
            return

        df = pd.read_csv(self.cfg.pheno_file)
        if self.cfg.sample_id_col not in df.columns:
            warnings.warn(f"表型组文件中未找到列: {self.cfg.sample_id_col}")
            self._pheno_features = pd.DataFrame(index=common_ids)
            return

        # 设置索引，移除ID列，只保留公共样本
        df = df.set_index(self.cfg.sample_id_col)
        df = df.loc[df.index.isin(common_ids)]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self._pheno_features = df[numeric_cols].loc[common_ids]

    # ------------------------------------------------------------------
    # 特征整合
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 变异评分与 top-k 选择
    # ------------------------------------------------------------------

    @staticmethod
    def _diff_vectors(emb_dict: dict):
        dh1 = np.asarray(emb_dict.get("Mut_hap1", []), dtype=float) - \
              np.asarray(emb_dict.get("WT_hap1", []), dtype=float)
        dh2 = np.asarray(emb_dict.get("Mut_hap2", []), dtype=float) - \
              np.asarray(emb_dict.get("WT_hap2", []), dtype=float)
        dref = np.asarray(emb_dict.get("Mut_ref", []), dtype=float) - \
               np.asarray(emb_dict.get("WT_ref", []), dtype=float)
        return dh1, dh2, dref

    @staticmethod
    def _cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _score_relative(dh1, dh2, dref):
        """策略1：偏离背景的程度，0=一致，2=完全相反。"""
        sim1 = MultiOmicsDataLoader._cosine_sim(dh1, dref)
        sim2 = MultiOmicsDataLoader._cosine_sim(dh2, dref)
        return 1.0 - (sim1 + sim2) / 2.0

    @staticmethod
    def _build_rep_vector(emb_dict: dict, strategy: str):
        """根据表示策略构建变异 embedding 向量。

        策略：
          mut_hap1:             仅 Mut_hap1
          mut_hap2:             仅 Mut_hap2
          mut_hap1_hap2_mean:   (Mut_hap1 + Mut_hap2) / 2
          mut_hap1_hap2_concat: concat(Mut_hap1, Mut_hap2)
          mut_mean_minus_wt_ref: (Mut_hap1 + Mut_hap2) / 2 - WT_ref
          mut_concat_minus_wt_ref: concat(Mut_hap1 - WT_ref, Mut_hap2 - WT_ref)
          baseline_delta_deltaref: Concat(Emb_baseline, Emb_delta, Emb_delta_ref)
            维度 = 3 * hidden_dim
        """
        mh1 = np.asarray(emb_dict.get("Mut_hap1", []), dtype=float)
        if len(mh1) == 0:
            return None
        dim = len(mh1)

        if strategy == "mut_hap1":
            return mh1

        mh2 = np.asarray(emb_dict.get("Mut_hap2", []), dtype=float)
        wt_ref = np.asarray(emb_dict.get("WT_ref", []), dtype=float)

        if strategy == "mut_hap2":
            return mh2 if len(mh2) == dim else mh1

        has_mh2 = len(mh2) == dim
        has_wt = len(wt_ref) == dim

        if strategy == "mut_hap1_hap2_mean":
            if not has_mh2:
                return mh1
            return (mh1 + mh2) / 2.0

        if strategy == "mut_hap1_hap2_concat":
            if not has_mh2:
                return mh1
            return np.hstack([mh1, mh2])

        if strategy == "mut_mean_minus_wt_ref":
            mean_vec = (mh1 + mh2) / 2.0 if has_mh2 else mh1
            return mean_vec - wt_ref if has_wt else mean_vec

        if strategy == "mut_concat_minus_wt_ref":
            left = mh1 - wt_ref if has_wt else mh1
            right = mh2 - wt_ref if has_wt and has_mh2 else (mh2 if has_mh2 else mh1)
            return np.hstack([left, right])

        if strategy == "baseline_delta_deltaref":
            wh1 = np.asarray(emb_dict.get("WT_hap1", []), dtype=float)
            wh2 = np.asarray(emb_dict.get("WT_hap2", []), dtype=float)
            mr = np.asarray(emb_dict.get("Mut_ref", []), dtype=float)
            wr = np.asarray(emb_dict.get("WT_ref", []), dtype=float)

            has_wh1 = len(wh1) == dim
            has_wh2 = len(wh2) == dim
            has_full_hap2 = has_mh2 and has_wh2
            has_ref = len(mr) == dim and len(wr) == dim

            # Emb_baseline = Mean(WT_hap1, WT_hap2)
            if has_wh1 and has_wh2:
                baseline = (wh1 + wh2) / 2.0
            elif has_wh1:
                baseline = wh1
            else:
                baseline = np.zeros(dim)

            # Emb_delta = Mean(Mut_hap1-WT_hap1, Mut_hap2-WT_hap2)
            delta_h1 = mh1 - wh1 if has_wh1 else np.zeros(dim)
            if has_full_hap2:
                delta = (delta_h1 + (mh2 - wh2)) / 2.0
            else:
                delta = delta_h1

            # Emb_delta_ref = Mut_ref - WT_ref
            delta_ref = (mr - wr) if has_ref else np.zeros(dim)

            return np.hstack([baseline, delta, delta_ref])

        # 未知策略 → fallback
        return mh1

    @staticmethod
    def _score_absolute(dh1, dh2, dref):
        """策略2：变异绝对效应大小（含 ref 背景效应）。"""
        return float(np.linalg.norm(dh1) + np.linalg.norm(dh2) + np.linalg.norm(dref))

    @staticmethod
    def _score_delta(emb_dict: dict):
        """策略：||Emb_delta||_2，即变异效应向量的 L2 范数。"""
        mh1 = np.asarray(emb_dict.get("Mut_hap1", []), dtype=float)
        wh1 = np.asarray(emb_dict.get("WT_hap1", []), dtype=float)
        if len(mh1) == 0:
            return 0.0
        dim = len(mh1)
        delta = mh1 - wh1 if len(wh1) == dim else mh1
        mh2 = np.asarray(emb_dict.get("Mut_hap2", []), dtype=float)
        wh2 = np.asarray(emb_dict.get("WT_hap2", []), dtype=float)
        if len(mh2) == dim and len(wh2) == dim:
            delta = (delta + (mh2 - wh2)) / 2.0
        return float(np.linalg.norm(delta))

    def _select_top_variants(self, variant_data: list) -> list:
        """
        对样本内所有变异评分，选取 top-k 的 Mut_hap1 向量。

        variant_data: [(emb_dict, mut_hap1_vec), ...]
        返回: top-k 的 mut_hap1 向量列表
        """
        top_k = self.cfg.top_k
        if top_k <= 0 or len(variant_data) <= top_k:
            return [v[1] for v in variant_data]

        strategy = self.cfg.variant_scoring
        n = len(variant_data)

        # 预计算原始分
        raw_rel = []
        raw_abs = []
        raw_delta = []
        for emb_dict, _ in variant_data:
            dh1, dh2, dref = self._diff_vectors(emb_dict)
            raw_rel.append(self._score_relative(dh1, dh2, dref))
            raw_abs.append(self._score_absolute(dh1, dh2, dref))
            raw_delta.append(self._score_delta(emb_dict))

        if strategy == "relative":
            final_scores = raw_rel
        elif strategy == "absolute":
            final_scores = raw_abs
        elif strategy == "delta":
            final_scores = raw_delta
        elif strategy == "weighted":
            final_scores = self._combine_weighted(raw_rel, raw_abs)
        elif strategy == "cascade":
            return self._select_cascade(raw_abs, raw_rel, variant_data, top_k)
        else:
            raise ValueError(f"未知 variant_scoring: {strategy!r}")

        # 按最终分降序取 top-k
        ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in ranked[:top_k]]
        return [variant_data[i][1] for i in top_indices]

    @staticmethod
    def _to_rank_pct(arr: np.ndarray) -> np.ndarray:
        """将数组转为 rank percentile [0, 1]，不受离群值和量级影响。"""
        n = len(arr)
        if n <= 1:
            return np.full(n, 0.5)
        order = np.argsort(arr)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        return ranks / (n - 1)

    def _combine_weighted(self, raw_rel, raw_abs):
        """策略3：rank percentile 归一化后加权求和（量级无关）。"""
        rel = np.asarray(raw_rel, dtype=float)
        abs_ = np.asarray(raw_abs, dtype=float)
        rel_pct = self._to_rank_pct(rel)
        abs_pct = self._to_rank_pct(abs_)
        w = self.cfg.score_weight
        return (w * rel_pct + (1 - w) * abs_pct).tolist()

    def _select_cascade(self, raw_abs, raw_rel, variant_data, top_k):
        """策略4：先用 absolute 粗筛 λk，再用 relative 精选 k。"""
        lamb = max(self.cfg.cascade_lambda, 1.0)
        pool_size = min(int(top_k * lamb), len(variant_data))
        pool_size = max(pool_size, top_k)
        ranked_abs = sorted(enumerate(raw_abs), key=lambda x: x[1], reverse=True)
        pool_indices = [i for i, _ in ranked_abs[:pool_size]]
        ranked_rel = sorted(pool_indices, key=lambda i: raw_rel[i], reverse=True)
        top_indices = ranked_rel[:top_k]
        return [variant_data[i][1] for i in top_indices]


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
