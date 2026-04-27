"""
apps.ml.ablator
================
消融实验管理器。

用于评估不同模态组合对分类性能的影响。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import MultiOmicsDataLoader
from .models import AVAILABLE_MODELS
from .trainer import MLTrainer
from .evaluator import MLEvaluator


# 模态映射：消融配置 -> 实际启用模块
MODULE_MAPPING = {
    "genome_only": ["genome"],
    "metab_only": ["metab"],
    "pheno_only": ["pheno"],
    "genome+metab": ["genome", "metab"],
    "genome+pheno": ["genome", "pheno"],
    "metab+pheno": ["metab", "pheno"],
    "all": ["genome", "metab", "pheno"],
}

# 人类可读的模态名称
MODULE_DISPLAY_NAMES = {
    "genome_only": "Genome (Embedding)",
    "metab_only": "Metabolome",
    "pheno_only": "Phenome",
    "genome+metab": "Genome + Metabolome",
    "genome+pheno": "Genome + Phenome",
    "metab+pheno": "Metabolome + Phenome",
    "all": "All Modalities",
}


class AblationStudy:
    """
    消融实验管理器。

    用法：
        ablator = AblationStudy(config, data_loader)
        results = ablator.run()
        ablator.save_results()
        ablator.print_summary()
    """

    def __init__(
        self,
        ablation_cfg,
        cv_cfg,
        hyperparam_cfg,
        output_cfg,
        preprocess_cfg=None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            ablation_cfg: AblationConfig实例
            cv_cfg: CVConfig实例
            hyperparam_cfg: HyperparamConfig实例
            output_cfg: OutputConfig实例
            preprocess_cfg: PreprocessConfig实例（可选）
            random_state: 随机种子
            n_jobs: 并行任务数
        """
        self.ablation_cfg = ablation_cfg
        self.cv_cfg = cv_cfg
        self.hyperparam_cfg = hyperparam_cfg
        self.output_cfg = output_cfg
        self.preprocess_cfg = preprocess_cfg
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.results = {}
        self.best_overall = None

    def run(
        self,
        data_loader: MultiOmicsDataLoader,
        model_names: Optional[List[str]] = None,
        scoring: str = "roc_auc",
        verbose: int = 1,
    ) -> Dict:
        """
        运行消融实验。

        Args:
            data_loader: MultiOmicsDataLoader实例
            model_names: 要测试的模型列表
            scoring: 评分指标
            verbose: 输出详细程度

        Returns:
            消融实验结果
        """
        if model_names is None:
            model_names = ["svm", "logistic_regression", "xgboost", "mlp"]

        modules = self.ablation_cfg.modules
        save_dir = Path(self.ablation_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        for module in modules:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"Ablation module: {MODULE_DISPLAY_NAMES.get(module, module)}")
                print(f"{'#'*60}")

            # 加载对应模态的数据
            try:
                modules_list = MODULE_MAPPING.get(module, [module])
                X, y, feature_names, sample_ids = data_loader.load_subset(modules_list)
            except Exception as e:
                if verbose:
                    print(f"  Data load failed: {e}")
                continue

            if verbose:
                print(f"  数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
                print(f"  标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

            # 训练所有模型
            trainer = MLTrainer(
                self.cv_cfg, self.output_cfg,
                preprocess_cfg=self.preprocess_cfg,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

            module_results = trainer.evaluate_all_models(
                X, y,
                model_names=model_names,
                param_grids=self.hyperparam_cfg.__dict__,
                scoring=scoring,
                verbose=verbose,
                feature_names=feature_names,
            )

            # 保存结果
            for model_name, results in module_results.items():
                trainer.save_results(results, module, model_name)

            all_results[module] = module_results

        self.results = all_results

        # 找出最佳组合
        self._find_best()

        return all_results

    def _find_best(self):
        """找出最佳模态+模型组合"""
        best_score = 0
        best_module = None
        best_model = None

        for module, model_results in self.results.items():
            for model_name, results in model_results.items():
                score = results.get("mean_score", 0)
                if score > best_score:
                    best_score = score
                    best_module = module
                    best_model = model_name

        self.best_overall = {
            "module": best_module,
            "model": best_model,
            "score": best_score,
            "params": self.results.get(best_module, {}).get(best_model, {}).get("best_params", {}),
        }

    def save_results(self):
        """保存所有消融实验结果"""
        save_dir = Path(self.ablation_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存汇总表
        summary = self.get_summary_table()
        summary_file = save_dir / "ablation_summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary table saved: {summary_file}")

        # 保存最佳配置
        if self.best_overall:
            best_file = save_dir / "best_config.json"
            with open(best_file, "w") as f:
                json.dump(self.best_overall, f, indent=2)
            print(f"Best config saved: {best_file}")

        # 保存完整结果
        full_results_file = save_dir / "full_ablation_results.json"
        with open(full_results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Full results saved: {full_results_file}")

    def get_summary_table(self) -> pd.DataFrame:
        """
        获取消融实验汇总表。

        Returns:
            DataFrame，包含每个模态+模型的CV分数
        """
        rows = []
        for module, model_results in self.results.items():
            for model_name, results in model_results.items():
                rows.append({
                    "module": MODULE_DISPLAY_NAMES.get(module, module),
                    "module_key": module,
                    "model": model_name,
                    "cv_score": results.get("mean_score", 0),
                    "cv_std": results.get("std_score", 0),
                    "best_params": str(results.get("best_params", {})),
                    "training_time": results.get("training_time", 0),
                })

        df = pd.DataFrame(rows)
        df = df.sort_values("cv_score", ascending=False)
        return df

    def print_summary(self):
        """打印消融实验摘要"""
        if not self.results:
            print("No ablation study results")
            return

        print(f"\n{'='*80}")
        print("Ablation Study Results Summary")
        print(f"{'='*80}")

        summary = self.get_summary_table()
        print(summary.to_string(index=False))

        if self.best_overall:
            print(f"\n🏆 Best combination:")
            print(f"   Modality: {MODULE_DISPLAY_NAMES.get(self.best_overall['module'], self.best_overall['module'])}")
            print(f"   Model: {self.best_overall['model']}")
            print(f"   CV score: {self.best_overall['score']:.4f}")
            print(f"   Params: {self.best_overall['params']}")

        print(f"{'='*80}")

    def get_module_comparison(self) -> pd.DataFrame:
        """
        获取按模态分组的性能比较（聚合所有模型）。

        Returns:
            DataFrame，每个模态的平均/最佳性能
        """
        rows = []
        for module, model_results in self.results.items():
            scores = [r.get("mean_score", 0) for r in model_results.values()]
            rows.append({
                "module": MODULE_DISPLAY_NAMES.get(module, module),
                "module_key": module,
                "best_cv_score": max(scores) if scores else 0,
                "mean_cv_score": np.mean(scores) if scores else 0,
                "std_cv_score": np.std(scores) if len(scores) > 1 else 0,
                "n_models": len(scores),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("best_cv_score", ascending=False)
        return df

    def plot_comparison(self, save_path: Optional[str] = None):
        """
        绘制消融实验对比图。

        Args:
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过绘图")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1：按模态对比
        module_df = self.get_module_comparison()
        ax1 = axes[0]
        ax1.barh(module_df["module"], module_df["best_cv_score"])
        ax1.set_xlabel("Best CV Score (ROC-AUC)")
        ax1.set_title("Performance by Modality")
        ax1.invert_yaxis()

        # 图2：热力图（模态 x 模型）
        ax2 = axes[1]
        heatmap_data = []
        modules = list(self.results.keys())
        models = []
        if modules:
            models = list(self.results[modules[0]].keys())

        for module in modules:
            row = []
            for model in models:
                score = self.results[module].get(model, {}).get("mean_score", 0)
                row.append(score)
            heatmap_data.append(row)

        if heatmap_data:
            im = ax2.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45)
            ax2.set_yticks(range(len(modules)))
            ax2.set_yticklabels([MODULE_DISPLAY_NAMES.get(m, m) for m in modules])
            ax2.set_title("Modality x Model Performance Heatmap")
            plt.colorbar(im, ax=ax2, label="CV Score")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved: {save_path}")
        else:
            plt.show()

        plt.close()
