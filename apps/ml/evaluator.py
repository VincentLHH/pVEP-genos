"""
apps.ml.evaluator
==================
ML评估器：计算各类评估指标。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


class MLEvaluator:
    """
    ML模型评估器。

    计算指标：
    - 分类指标：ACC, Precision, Recall, F1
    - 排序指标：AUC-ROC, AUC-PR
    - 混淆矩阵
    """

    def __init__(self, output_dir: str = "outputs/ml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        pos_label: int = 1,
    ) -> Dict:
        """
        评估预测结果。

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选，用于计算AUC）
            pos_label: 正类标签

        Returns:
            评估指标字典
        """
        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        }

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
        results["tn"] = int(cm[0, 0])
        results["fp"] = int(cm[0, 1])
        results["fn"] = int(cm[1, 0])
        results["tp"] = int(cm[1, 1])

        # AUC指标
        if y_proba is not None:
            try:
                results["auc_roc"] = float(self._safe_auc(y_true, y_proba))
                results["auc_pr"] = float(average_precision_score(y_true, y_proba))
            except ValueError:
                results["auc_roc"] = None
                results["auc_pr"] = None

        return results

    def _safe_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """安全计算AUC，处理所有样本同类别的情况"""
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)

    def evaluate_cv_results(
        self,
        y_true: np.ndarray,
        oof_pred: np.ndarray,
        oof_proba: np.ndarray,
    ) -> Dict:
        """
        评估交叉验证结果。

        Args:
            y_true: 真实标签
            oof_pred: OOF预测标签
            oof_proba: OOF预测概率

        Returns:
            评估指标字典
        """
        results = self.evaluate(y_true, oof_pred, oof_proba)

        # 分类报告
        results["classification_report"] = classification_report(
            y_true, oof_pred, output_dict=True, zero_division=0
        )

        # ROC曲线数据
        if oof_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, oof_proba)
            results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        # PR曲线数据
        if oof_proba is not None:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, oof_proba)
            results["pr_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else [],
            }

        return results

    def compare_models(
        self,
        cv_results: Dict[str, Dict],
    ) -> List[Dict]:
        """
        比较多个模型的CV结果。

        Args:
            cv_results: {model_name: cv_result} 字典

        Returns:
            按性能排序的模型列表
        """
        comparisons = []

        for model_name, results in cv_results.items():
            comparison = {
                "model": model_name,
                "cv_score": results.get("mean_score", 0),
                "cv_std": results.get("std_score", 0),
                "best_params": results.get("best_params", {}),
            }

            # 添加评估指标
            if "oof_predictions" in results:
                oof_pred = np.array(results["oof_predictions"])
                oof_proba = np.array(results.get("oof_probabilities", []))
                eval_results = self.evaluate_cv_results(
                    np.array(results.get("y_true", [])), oof_pred, oof_proba
                )
                comparison.update(eval_results)

            comparisons.append(comparison)

        # 按CV分数排序
        comparisons.sort(key=lambda x: x["cv_score"], reverse=True)

        return comparisons

    def save_evaluation_report(
        self,
        results: Dict,
        module_name: str,
        model_name: str,
    ):
        """保存评估报告"""
        report = {
            "module": module_name,
            "model": model_name,
            "metrics": results,
        }

        report_file = self.output_dir / f"{module_name}_{model_name}_evaluation.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def print_summary(self, results: Dict, module_name: str, model_name: str):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"评估结果: {module_name} - {model_name}")
        print(f"{'='*60}")
        print(f"  Accuracy:  {results.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {results.get('precision', 'N/A'):.4f}")
        print(f"  Recall:    {results.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score:  {results.get('f1', 'N/A'):.4f}")
        if results.get("auc_roc"):
            print(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        if results.get("auc_pr"):
            print(f"  AUC-PR:    {results['auc_pr']:.4f}")
        print(f"{'='*60}")
