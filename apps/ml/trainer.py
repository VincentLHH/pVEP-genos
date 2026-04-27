"""
apps.ml.trainer
================
ML训练器：交叉验证 + 超参搜索。
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from .models import create_model, get_param_grid, AVAILABLE_MODELS


class MLTrainer:
    """
    机器学习训练器。

    功能：
    1. 预处理（集成到CV中，防止数据泄露）
    2. 交叉验证评估
    3. 超参搜索（GridSearchCV）
    4. 模型训练与预测

    用法：
        trainer = MLTrainer(config)
        results = trainer.cv_evaluate(X, y, model_name="xgboost")
        best_model = trainer.train(X, y, model_name="xgboost", params=best_params)
    """

    def __init__(
        self,
        cv_cfg,
        output_cfg,
        preprocess_cfg=None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            cv_cfg: CVConfig实例
            output_cfg: OutputConfig实例
            preprocess_cfg: PreprocessConfig实例（可选）
            random_state: 随机种子
            n_jobs: 并行任务数
        """
        self.cv_cfg = cv_cfg
        self.output_cfg = output_cfg
        self.preprocess_cfg = preprocess_cfg
        self.random_state = random_state
        self.n_jobs = n_jobs

        # 创建交叉验证器
        self.cv = StratifiedKFold(
            n_splits=cv_cfg.n_folds,
            shuffle=cv_cfg.shuffle,
            random_state=random_state,
        )

        # 初始化预处理器（延迟）
        self._preprocessor = None

    def _get_preprocessor(self):
        if self.preprocess_cfg is None or not self.preprocess_cfg.enabled:
            return None

        from .preprocessor import MultiOmicsPreprocessor

        if self._preprocessor is None:
            self._preprocessor = MultiOmicsPreprocessor(
                emb_n_components=self.preprocess_cfg.emb_n_components,
                emb_standardize_first=self.preprocess_cfg.emb_standardize_first,
                tab_strategy=self.preprocess_cfg.tab_impute_strategy,
                random_state=self.random_state,
            )

        return self._preprocessor

    def _preprocess_fold(self, X_train, X_val, X_test=None, feature_names=None):
        preprocessor = self._get_preprocessor()
        if preprocessor is None:
            return (X_train, X_val) if X_test is None else (X_train, X_val, X_test)

        preprocessor.fit(X_train, feature_names=feature_names)

        X_train_proc = preprocessor.transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        if X_test is not None:
            X_test_proc = preprocessor.transform(X_test)
            return X_train_proc, X_val_proc, X_test_proc

        return X_train_proc, X_val_proc

    def cv_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        param_grid: Optional[List[Dict]] = None,
        scoring: str = "roc_auc",
        verbose: int = 1,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        交叉验证评估。

        预处理在每个fold的train部分fit，确保无数据泄露。

        Args:
            X: 特征矩阵
            y: 标签向量
            model_name: 模型名
            param_grid: 超参网格（None则使用默认/配置的网格）
            scoring: 评分指标

        Returns:
            包含CV结果的字典
        """
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"模型: {model_name}")
            print(f"数据: X.shape={X.shape}, y分布={dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"{'='*60}")

        results = {
            "model": model_name,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "cv_folds": self.cv_cfg.n_folds,
            "scoring": scoring,
            "fold_scores": [],
            "mean_score": 0.0,
            "std_score": 0.0,
            "best_params": None,
            "training_time": 0.0,
        }

        # 手动CV循环（支持预处理）
        # 使用手动循环以确保预处理在每个fold内正确fit/transform
        oof_preds = np.zeros(len(y))
        oof_probas = np.zeros(len(y))
        fold_scores_list = []
        best_score = -1
        best_params = None

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 在fold内预处理（仅首 fold 用 feature_names 推断列索引）
            X_train_proc, X_val_proc = self._preprocess_fold(
                X_train, X_val, feature_names=feature_names)
            results["n_features"] = X_train_proc.shape[1]  # 更新处理后特征数

            if param_grid and len(param_grid) > 1:
                # 超参搜索：在此fold内搜索最佳参数
                fold_best_score = -1
                fold_best_params = None
                fold_best_model = None

                for params in param_grid:
                    model = create_model(model_name, random_state=self.random_state, **params)
                    model.fit(X_train_proc, y_train)

                    if scoring == "roc_auc":
                        score = roc_auc_score(y_val, model.predict_proba(X_val_proc)[:, 1])
                    else:
                        score = model.score(X_val_proc, y_val)

                    if score > fold_best_score:
                        fold_best_score = score
                        fold_best_params = params
                        fold_best_model = model

                fold_scores_list.append(fold_best_score)
                # OOF预测用当前fold最佳模型
                oof_preds[val_idx] = fold_best_model.predict(X_val_proc)
                oof_probas[val_idx] = fold_best_model.predict_proba(X_val_proc)[:, 1]

                if fold_best_score > best_score:
                    best_score = fold_best_score
                    best_params = fold_best_params

            else:
                # 无超参搜索
                if param_grid:
                    params = param_grid[0] if param_grid else {}
                else:
                    params = {}

                model = create_model(model_name, random_state=self.random_state, **params)
                model.fit(X_train_proc, y_train)

                if scoring == "roc_auc":
                    fold_score = roc_auc_score(y_val, model.predict_proba(X_val_proc)[:, 1])
                else:
                    fold_score = model.score(X_val_proc, y_val)

                fold_scores_list.append(fold_score)
                # OOF预测用当前fold模型
                oof_preds[val_idx] = model.predict(X_val_proc)
                oof_probas[val_idx] = model.predict_proba(X_val_proc)[:, 1]

                if fold_score > best_score:
                    best_score = fold_score
                    best_params = params

            if verbose:
                print(f"  Fold {fold_idx + 1}: {scoring}={fold_scores_list[-1]:.4f}")

        # 汇总结果
        results["fold_scores"] = fold_scores_list
        results["mean_score"] = float(np.mean(fold_scores_list))
        results["std_score"] = float(np.std(fold_scores_list))
        results["best_params"] = best_params
        results["preprocessing"] = {
            "enabled": self.preprocess_cfg is not None and self.preprocess_cfg.enabled,
            "emb_n_components": str(self.preprocess_cfg.emb_n_components) if self.preprocess_cfg else "N/A",
            "tab_impute_strategy": self.preprocess_cfg.tab_impute_strategy if self.preprocess_cfg else "N/A",
        }

        if verbose:
            print(f"CV分数: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            print(f"最佳参数: {best_params}")

        # 使用全部数据训练最终模型
        if self._preprocessor is not None:
            self._preprocessor.fit(X, feature_names=feature_names)
            X_proc = self._preprocessor.transform(X)
        else:
            X_proc = X

        final_model = create_model(model_name, random_state=self.random_state, **best_params)
        final_model.fit(X_proc, y)
        results["training_time"] = time.time() - start_time

        # OOF预测结果
        results["oof_predictions"] = oof_preds.tolist()
        results["oof_probabilities"] = oof_probas.tolist()
        results["y_true"] = y.tolist()

        # 计算额外指标
        if len(np.unique(y)) == 2:
            results["oof_auc"] = float(roc_auc_score(y, oof_probas))

        if verbose:
            print(f"耗时: {results['training_time']:.1f}s")

        return results

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        params: Optional[Dict] = None,
    ):
        """
        使用全部数据训练模型。

        Args:
            X: 特征矩阵
            y: 标签向量
            model_name: 模型名
            params: 模型参数

        Returns:
            训练好的模型
        """
        params = params or {}
        model = create_model(model_name, random_state=self.random_state, **params)
        model.fit(X, y)
        return model

    def evaluate_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None,
        param_grids: Optional[Dict[str, List[Dict]]] = None,
        scoring: str = "roc_auc",
        verbose: int = 1,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        if model_names is None:
            model_names = AVAILABLE_MODELS
        if param_grids is None:
            param_grids = {}

        all_results: Dict[str, Dict] = {}

        for model_name in model_names:
            if param_grids and model_name in param_grids:
                param_grid = get_param_grid(model_name, param_grids)
            else:
                param_grid = None
            results = self.cv_evaluate(
                X, y, model_name,
                param_grid=param_grid,
                scoring=scoring,
                verbose=verbose,
                feature_names=feature_names,
            )
            all_results[model_name] = results

        # 排序输出
        if verbose:
            print(f"\n{'='*60}")
            print("模型排名（按CV分数）:")
            print(f"{'='*60}")
            sorted_models = sorted(
                all_results.items(),
                key=lambda x: x[1]["mean_score"],
                reverse=True
            )
            for rank, (name, res) in enumerate(sorted_models, 1):
                print(f"  {rank}. {name}: {res['mean_score']:.4f} (+/- {res.get('std_score', 0):.4f})")

        return all_results

    def save_results(self, results: Dict, module_name: str, model_name: str):
        """保存训练结果"""
        save_dir = Path(self.output_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存CV结果
        if self.output_cfg.save_best_params:
            params_file = save_dir / f"{module_name}_{model_name}_params.json"
            with open(params_file, "w") as f:
                json.dump(results.get("best_params", {}), f, indent=2)

        # 保存OOF预测
        if self.output_cfg.save_predictions:
            pred_file = save_dir / f"{module_name}_{model_name}_predictions.json"
            with open(pred_file, "w") as f:
                json.dump({
                    "oof_predictions": results.get("oof_predictions", []),
                    "oof_probabilities": results.get("oof_probabilities", []),
                }, f, indent=2)

        # 保存完整CV结果
        cv_file = save_dir / f"{module_name}_{model_name}_cv_results.json"
        with open(cv_file, "w") as f:
            # 移除不可序列化的对象
            serializable_results = {k: v for k, v in results.items()
                                    if k not in ["best_estimator_", "cv_results"]}
            json.dump(serializable_results, f, indent=2, default=str)

    def get_feature_importance(
        self,
        model,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """获取特征重要性"""
        importances = model.feature_importances_
        if importances is None:
            return []

        return sorted(
            zip(feature_names, importances),
            key=lambda x: abs(x[1]),
            reverse=True
        )
