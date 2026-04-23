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

from .models import create_model, get_param_grid, AVAILABLE_MODELS


class MLTrainer:
    """
    机器学习训练器。

    功能：
    1. 交叉验证评估
    2. 超参搜索（GridSearchCV）
    3. 模型训练与预测

    用法：
        trainer = MLTrainer(config)
        results = trainer.cv_evaluate(X, y, model_name="xgboost")
        best_model = trainer.train(X, y, model_name="xgboost", params=best_params)
    """

    def __init__(self, cv_cfg, output_cfg, random_state: int = 42, n_jobs: int = -1):
        """
        Args:
            cv_cfg: CVConfig实例
            output_cfg: OutputConfig实例
            random_state: 随机种子
            n_jobs: 并行任务数
        """
        self.cv_cfg = cv_cfg
        self.output_cfg = output_cfg
        self.random_state = random_state
        self.n_jobs = n_jobs

        # 创建交叉验证器
        self.cv = StratifiedKFold(
            n_splits=cv_cfg.n_folds,
            shuffle=cv_cfg.shuffle,
            random_state=random_state,
        )

    def cv_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        param_grid: Optional[List[Dict]] = None,
        scoring: str = "roc_auc",
        verbose: int = 1,
    ) -> Dict:
        """
        交叉验证评估。

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

        # 超参搜索
        if param_grid and len(param_grid) > 1:
            if verbose:
                print(f"超参搜索: {len(param_grid)} 个组合")

            model = create_model(model_name, random_state=self.random_state)
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=verbose,
                return_train_score=True,
            )
            grid_search.fit(X, y)

            results["best_params"] = grid_search.best_params_
            results["mean_score"] = grid_search.best_score_
            results["cv_results"] = {
                "mean_test_score": float(np.mean(grid_search.cv_results_["mean_test_score"])),
                "std_test_score": float(np.std(grid_search.cv_results_["mean_test_score"])),
                "n_combinations": len(param_grid),
            }

            if verbose:
                print(f"最佳参数: {grid_search.best_params_}")
                print(f"CV分数: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_score']:.4f})")

            best_model = grid_search.best_estimator_
        else:
            # 单参数或无超参搜索
            if param_grid:
                params = param_grid[0] if param_grid else {}
            else:
                params = {}

            model = create_model(model_name, random_state=self.random_state, **params)

            fold_scores = cross_val_score(
                model, X, y, cv=self.cv, scoring=scoring, n_jobs=self.n_jobs
            )

            results["fold_scores"] = fold_scores.tolist()
            results["mean_score"] = float(np.mean(fold_scores))
            results["std_score"] = float(np.std(fold_scores))
            results["best_params"] = params

            if verbose:
                print(f"CV分数: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
                print(f"各折分数: {[f'{s:.4f}' for s in fold_scores]}")

            # 使用全部数据训练最终模型
            model.fit(X, y)
            best_model = model

        results["training_time"] = time.time() - start_time

        # 获取OOF预测
        try:
            oof_proba = cross_val_predict(
                best_model, X, y, cv=self.cv, method="predict_proba", n_jobs=self.n_jobs
            )[:, 1]
            oof_pred = (oof_proba > 0.5).astype(int)
            results["oof_predictions"] = oof_pred.tolist()
            results["oof_probabilities"] = oof_proba.tolist()

            # 计算额外指标
            if len(np.unique(y)) == 2:
                results["oof_auc"] = float(roc_auc_score(y, oof_proba))
        except Exception:
            pass

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
    ) -> Dict[str, Dict]:
        """
        评估所有模型。

        Args:
            X: 特征矩阵
            y: 标签向量
            model_names: 要评估的模型列表（None则评估全部）
            param_grids: 各模型的超参网格
            scoring: 评分指标

        Returns:
            {model_name: results} 字典
        """
        if model_names is None:
            model_names = AVAILABLE_MODELS
        if param_grids is None:
            param_grids = {}

        all_results = {}

        for model_name in model_names:
            param_grid = param_grids.get(model_name)
            results = self.cv_evaluate(
                X, y, model_name,
                param_grid=param_grid,
                scoring=scoring,
                verbose=verbose,
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
