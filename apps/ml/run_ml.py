#!/usr/bin/env python
"""
apps.ml.run_ml
==============
ML应用主入口。

用法：
    # 完整运行（超参搜索 + 消融实验）
    python apps/ml/run_ml.py --config apps/configs/default_ml.yaml

    # 只跑消融实验（使用已有的最佳参数）
    python apps/ml/run_ml.py --config apps/configs/default_ml.yaml \
        --ablation-only --best-params outputs/ablation/best_config.json

    # 只测试单个模态
    python apps/ml/run_ml.py --config apps/configs/default_ml.yaml \
        --modules genome_only --models xgboost

    # 跳过超参搜索（使用默认参数）
    python apps/ml/run_ml.py --config apps/configs/default_ml.yaml \
        --no-hyperparam-search
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.ml.config import (
    load_config,
    DataConfig,
    CVConfig,
    HyperparamConfig,
    AblationConfig,
    OutputConfig,
    GlobalConfig,
)
from apps.ml.data_loader import MultiOmicsDataLoader
from apps.ml.models import AVAILABLE_MODELS
from apps.ml.trainer import MLTrainer
from apps.ml.evaluator import MLEvaluator
from apps.ml.ablator import AblationStudy


def parse_args():
    parser = argparse.ArgumentParser(
        description="pVEP-genos ML分类应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="apps/configs/default_ml.yaml",
        help="配置文件路径"
    )

    # 消融实验选项
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="只运行消融实验（跳过超参搜索）"
    )
    parser.add_argument(
        "--best-params",
        type=str,
        help="已有的最佳参数JSON文件"
    )

    # 模态和模型选择
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["genome_only", "metab_only", "pheno_only",
                 "genome+metab", "genome+pheno", "metab+pheno", "all"],
        help="要测试的模态组合"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        help="要测试的模型"
    )

    # 超参搜索
    parser.add_argument(
        "--no-hyperparam-search",
        action="store_true",
        help="跳过超参搜索，使用默认参数"
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录（覆盖配置）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="输出详细程度"
    )

    return parser.parse_args()


def setup_random_seed(seed: int):
    """设置所有随机种子"""
    import random
    import os
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_full_pipeline(cfg: GlobalConfig, args):
    """完整流水线：数据加载 -> 超参搜索 -> 消融实验"""
    print(f"\n{'#'*80}")
    print("# pVEP-genos ML分类应用")
    print(f"{'#'*80}")

    # 设置随机种子
    setup_random_seed(cfg.random_state)
    print(f"随机种子: {cfg.random_state}")

    # ========== 1. 数据加载 ==========
    print(f"\n{'='*60}")
    print("1. 加载数据...")
    print(f"{'='*60}")

    data_loader = MultiOmicsDataLoader(cfg.data)

    # 先加载全部数据验证
    try:
        X, y, feature_names, sample_ids = data_loader.load_all()
        print(f"  样本数: {X.shape[0]}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  标签分布: {dict(zip(*zip(*[[k, v] for k, v in zip(*zip(*[[str(k), v] for k, v in zip(*np.unique(y, return_counts=True))])])]))) if hasattr(y, '__iter__') else {}}")
        print(f"  模态: genome={X.shape[1] if 'genome' in str(data_loader._emb_features.shape) else 0}, "
              f"metab={data_loader._metab_features.shape[1] if hasattr(data_loader._metab_features, 'shape') else 0}, "
              f"pheno={data_loader._pheno_features.shape[1] if hasattr(data_loader._pheno_features, 'shape') else 0}")
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return

    # ========== 2. 全部数据评估（获取最佳模型配置）==========
    if not args.no_hyperparam_search:
        print(f"\n{'='*60}")
        print("2. 超参搜索（全部模态）...")
        print(f"{'='*60}")

        trainer = MLTrainer(
            cfg.cv, cfg.output,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

        all_results = trainer.evaluate_all_models(
            X, y,
            model_names=args.models if args.models else AVAILABLE_MODELS,
            param_grids=cfg.hyperparam.__dict__,
            scoring="roc_auc",
            verbose=args.verbose,
        )

        # 保存结果
        for model_name, results in all_results.items():
            trainer.save_results(results, "all", model_name)

        # 找出最佳模型
        best_model = max(all_results.items(), key=lambda x: x[1]["mean_score"])
        print(f"\n🏆 最佳模型: {best_model[0]} (CV={best_model[1]['mean_score']:.4f})")

        # 保存最佳配置
        best_config = {
            "model": best_model[0],
            "params": best_model[1].get("best_params", {}),
            "cv_score": best_model[1]["mean_score"],
        }
        best_config_file = Path(cfg.output.save_dir) / "best_config.json"
        best_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(best_config_file, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"最佳配置已保存: {best_config_file}")

    # ========== 3. 消融实验 ==========
    if args.modules:
        cfg.ablation.modules = args.modules

    print(f"\n{'='*60}")
    print("3. 消融实验...")
    print(f"{'='*60}")

    ablator = AblationStudy(
        cfg.ablation,
        cfg.cv,
        cfg.hyperparam,
        cfg.output,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    ablator.run(
        data_loader,
        model_names=args.models if args.models else AVAILABLE_MODELS,
        scoring="roc_auc",
        verbose=args.verbose,
    )

    ablator.save_results()
    ablator.print_summary()

    # 绘制对比图
    try:
        fig_path = Path(cfg.ablation.save_dir) / "ablation_comparison.png"
        ablator.plot_comparison(str(fig_path))
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


def run_ablation_only(cfg: GlobalConfig, args):
    """只运行消融实验（使用已有参数）"""
    print(f"\n{'#'*80}")
    print("# pVEP-genos 消融实验（快速模式）")
    print(f"{'#'*80}")

    setup_random_seed(cfg.random_state)

    # 加载已有参数
    if args.best_params:
        with open(args.best_params) as f:
            best_config = json.load(f)
        print(f"使用已有参数: {best_config}")

    # 加载数据
    data_loader = MultiOmicsDataLoader(cfg.data)
    X, y, feature_names, sample_ids = data_loader.load_all()

    # 运行消融实验
    ablator = AblationStudy(
        cfg.ablation,
        cfg.cv,
        cfg.hyperparam,
        cfg.output,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )

    ablator.run(
        data_loader,
        model_names=args.models if args.models else AVAILABLE_MODELS,
        scoring="roc_auc",
        verbose=args.verbose,
    )

    ablator.save_results()
    ablator.print_summary()


def main():
    args = parse_args()

    # 加载配置
    if not Path(args.config).exists():
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)

    # 命令行覆盖
    if args.output_dir:
        cfg.output.save_dir = args.output_dir
    if args.modules:
        cfg.ablation.modules = args.modules

    Path(cfg.output.save_dir).mkdir(parents=True, exist_ok=True)

    # 保存运行配置
    from apps.ml.config import save_config
    save_config(cfg, Path(cfg.output.save_dir) / "run_config.yaml")

    # 运行
    if args.ablation_only:
        run_ablation_only(cfg, args)
    else:
        run_full_pipeline(cfg, args)


if __name__ == "__main__":
    main()
