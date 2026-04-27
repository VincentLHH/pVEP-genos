"""
tests.test_ml_pipeline
======================
apps/ml/ 模块集成测试，使用模拟 Monk 风格数据。

覆盖：
- MultiOmicsDataLoader：数据加载、ID 对齐、模态子集
- create_model / get_param_grid：模型创建与参数网格展开
- MLTrainer：CV 评估、超参搜索、预处理
- MultiOmicsPreprocessor：fit/transform、防泄露
- AblationStudy：消融实验流程
- 端到端：从数据 -> 加载 -> 训练 -> 评估
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.ml.data_loader import MultiOmicsDataLoader
from apps.ml.models import create_model, get_param_grid, AVAILABLE_MODELS
from apps.ml.trainer import MLTrainer
from apps.ml.evaluator import MLEvaluator
from apps.ml.preprocessor import (
    MultiOmicsPreprocessor,
    EmbeddingReducer,
    TableImputer,
)
from apps.ml.config import (
    DataConfig, PreprocessConfig, CVConfig,
    HyperparamConfig, AblationConfig, OutputConfig,
    GlobalConfig, load_config,
)
from apps.ml.ablator import AblationStudy


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================
# Monk 风格模拟数据
# ============================================================
# Monk-1 规则：a1 == a2 or a5 == 1
# 6 个属性：a1(3) a2(3) a3(2) a4(3) a5(4) a6(2)
# 生成 124 个训练样本 + 432 个测试样本

def _generate_monk1_data(n_samples=200, seed=42):
    """生成符合 Monk-1 规则的分类数据。"""
    rng = np.random.RandomState(seed)
    n_values = [3, 3, 2, 3, 4, 2]  # 每个属性的取值数

    X = np.zeros((n_samples, 6), dtype=int)
    for i, nv in enumerate(n_values):
        X[:, i] = rng.randint(1, nv + 1, size=n_samples)

    # Monk-1 规则：(a1 == a2) or (a5 == 1)
    y = ((X[:, 0] == X[:, 1]) | (X[:, 4] == 1)).astype(int)

    return X, y, n_values


def _one_hot_encode(X, n_values):
    """对离散特征做 one-hot 编码。"""
    parts = []
    for i, nv in enumerate(n_values):
        col = X[:, i].astype(int)
        oh = np.zeros((len(col), nv), dtype=float)
        oh[np.arange(len(col)), col - 1] = 1.0
        parts.append(oh)
    return np.hstack(parts)


def _make_emb_features(X_oh, n_emb=32, seed=42):
    """基于 one-hot 特征生成模拟 embedding 向量。"""
    rng = np.random.RandomState(seed)
    n_samples = X_oh.shape[0]
    proj = rng.randn(X_oh.shape[1], n_emb).astype(float) / np.sqrt(X_oh.shape[1])
    emb = X_oh @ proj + rng.randn(n_samples, n_emb) * 0.1
    return emb


@pytest.fixture(scope="module")
def monk_data():
    """模块级 Monk 数据 fixture。"""
    X_raw, y, n_values = _generate_monk1_data(n_samples=100)
    X_oh = _one_hot_encode(X_raw, n_values)
    return X_raw, X_oh, y, n_values


@pytest.fixture(scope="module")
def monk_data_dir(monk_data, tmp_path_factory):
    """生成模拟文件目录结构，用于测试 MultiOmicsDataLoader。"""
    X_raw, X_oh, y, n_values = monk_data
    n_samples = len(y)
    n_metab = 10
    n_pheno = 4
    n_emb = 32

    base = tmp_path_factory.mktemp("monk_data")

    # 1. Embedding JSON 文件
    emb_dir = base / "emb"
    emb_dir.mkdir()
    emb_vecs = _make_emb_features(X_oh, n_emb=n_emb)
    for i in range(n_samples):
        sid = f"monk_{i:03d}"
        emb_vec = emb_vecs[i]
        data = {
            "sample_id": sid,
            "embeddings": {
                "chr1_100_A_T": {
                    "Genos-1.2B": {
                        "Mut_hap1": emb_vec.tolist(),
                        "WT_hap1": (emb_vec * 0.01).tolist(),
                        "Mut_hap2": (emb_vec * 0.9).tolist(),
                        "WT_hap2": (emb_vec * 0.01).tolist(),
                        "Mut_ref": (emb_vec * 0.1).tolist(),
                        "WT_ref": (emb_vec * 0.0).tolist(),
                    }
                }
            }
        }
        with open(emb_dir / f"{sid}.json", "w") as f:
            json.dump(data, f)

    # 2. 代谢组 CSV
    rng = np.random.RandomState(RANDOM_STATE)
    metab_rows = []
    for i in range(n_samples):
        sid = f"monk_{i:03d}"
        row = {"sample_id": sid}
        for j in range(n_metab):
            row[f"metab_{j}"] = rng.randn()
        metab_rows.append(row)
    pd.DataFrame(metab_rows).to_csv(base / "metab.csv", index=False)

    # 3. 表型组 CSV
    pheno_rows = []
    for i in range(n_samples):
        sid = f"monk_{i:03d}"
        row = {"sample_id": sid}
        for j in range(n_pheno):
            row[f"pheno_{j}"] = rng.randn()
        pheno_rows.append(row)
    pd.DataFrame(pheno_rows).to_csv(base / "pheno.csv", index=False)

    # 4. 标签 CSV
    label_rows = []
    for i in range(n_samples):
        sid = f"monk_{i:03d}"
        label_rows.append({"sample_id": sid, "label": int(y[i])})
    pd.DataFrame(label_rows).to_csv(base / "labels.csv", index=False)

    return base


@pytest.fixture(scope="module")
def data_cfg(monk_data_dir):
    """DataConfig fixture。"""
    return DataConfig(
        emb_dir=str(monk_data_dir / "emb"),
        metab_file=str(monk_data_dir / "metab.csv"),
        pheno_file=str(monk_data_dir / "pheno.csv"),
        label_file=str(monk_data_dir / "labels.csv"),
        sample_id_col="sample_id",
        label_col="label",
        emb_aggregation="mean",
    )


# ============================================================
# TestDataLoader
# ============================================================

class TestDataLoaderWithMonk:
    """使用 Monk 风格数据测试 MultiOmicsDataLoader。"""

    def test_load_all(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        assert X.shape[0] > 0
        assert len(y) == X.shape[0]
        assert len(feature_names) == X.shape[1]
        assert len(sample_ids) == X.shape[0]
        assert any(n.startswith("emb_") for n in feature_names)
        assert any(n.startswith("metab_") for n in feature_names)
        assert any(n.startswith("pheno_") for n in feature_names)

    def test_load_subset_genome_only(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_subset(["genome"])

        assert X.shape[0] > 0
        assert all(n.startswith("emb_") for n in feature_names)

    def test_load_subset_metab_only(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_subset(["metab"])

        assert X.shape[0] > 0
        assert all(n.startswith("metab_") for n in feature_names)

    def test_load_subset_all(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_subset(["genome", "metab", "pheno"])

        X_genome, _, fn_genome, _ = loader.load_subset(["genome"])
        assert X.shape[1] > X_genome.shape[1]

    def test_id_alignment_no_extra(self, data_cfg):
        """对齐后不应有不在交集里的样本。"""
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        emb_dir = Path(data_cfg.emb_dir)
        emb_ids = {f.stem for f in emb_dir.glob("*.json")}
        assert set(sample_ids).issubset(emb_ids)

    def test_label_values(self, data_cfg):
        """标签应只有 0 和 1。"""
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, _, _ = loader.load_all()
        assert set(np.unique(y)).issubset({0, 1})


# ============================================================
# TestModels
# ============================================================

class TestModelsWithMonk:
    """使用 Monk 数据测试模型创建和参数网格。"""

    def test_create_all_models(self, monk_data):
        _, X_oh, y, _ = monk_data
        for name in AVAILABLE_MODELS:
            try:
                model = create_model(name, random_state=RANDOM_STATE)
                model.fit(X_oh, y)
                pred = model.predict(X_oh)
                assert len(pred) == len(y)
            except ImportError:
                pytest.skip(f"{name} 不可用")

    def test_get_param_grid_expansion(self):
        hp = HyperparamConfig()
        for name in AVAILABLE_MODELS:
            grid = get_param_grid(name, hp.__dict__)
            assert isinstance(grid, list)
            assert len(grid) > 0
            assert isinstance(grid[0], dict)

    def test_param_grid_svm(self):
        hp = HyperparamConfig()
        grid = get_param_grid("svm", hp.__dict__)
        # C(5) * kernel(2) * gamma(2) = 20
        assert len(grid) == 20
        assert all("C" in g and "kernel" in g and "gamma" in g for g in grid)

    def test_param_grid_unknown_model(self):
        hp = HyperparamConfig()
        grid = get_param_grid("nonexistent", hp.__dict__)
        assert grid == [{}]

    def test_create_model_with_params(self, monk_data):
        _, X_oh, y, _ = monk_data
        model = create_model("svm", random_state=RANDOM_STATE, C=1.0, kernel="linear")
        model.fit(X_oh, y)
        assert model.predict(X_oh).shape == y.shape

    def test_create_model_with_bad_params_ignored(self, monk_data):
        """create_model 应忽略不属于该模型的参数。"""
        _, X_oh, y, _ = monk_data
        # 传一个 SVM 不认识的参数，不应报错
        model = create_model("svm", random_state=RANDOM_STATE, C=1.0, nonexistent_param=999)
        model.fit(X_oh, y)
        assert model.predict(X_oh).shape == y.shape


# ============================================================
# TestPreprocessor
# ============================================================

class TestPreprocessorWithMonk:
    """使用 Monk 数据测试预处理器。"""

    def test_fit_transform(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        proc = MultiOmicsPreprocessor(emb_n_components=10, tab_strategy="median")
        proc.fit(X, feature_names)
        X_out = proc.transform(X)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] < X.shape[1]  # PCA 降维后列数应减少

    def test_no_leakage(self, monk_data):
        _, X_oh, y, _ = monk_data
        feature_names = [f"emb_{i}" for i in range(X_oh.shape[1])]
        proc = MultiOmicsPreprocessor(emb_n_components=5)
        proc.fit(X_oh[:50], feature_names)

        X_train_out = proc.transform(X_oh[:50])
        X_test_out = proc.transform(X_oh[50:])
        assert X_train_out.shape[1] == 5
        assert X_test_out.shape[1] == 5

    def test_emb_reducer(self, monk_data):
        _, X_oh, y, _ = monk_data
        reducer = EmbeddingReducer(n_components=5)
        X_out = reducer.fit_transform(X_oh)
        assert X_out.shape == (X_oh.shape[0], 5)

    def test_table_imputer(self):
        X = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]], dtype=float)
        imputer = TableImputer(strategy="median")
        X_out = imputer.fit_transform(X)
        assert not np.any(np.isnan(X_out))

    def test_preprocessor_auto_infer_columns(self, monk_data):
        """无 feature_names 时应自动推断列索引。"""
        _, X_oh, y, _ = monk_data
        proc = MultiOmicsPreprocessor(emb_n_components=5, tab_strategy="median")
        proc.fit(X_oh)  # 不传 feature_names
        X_out = proc.transform(X_oh)
        assert X_out.shape[0] == X_oh.shape[0]
        assert X_out.shape[1] < X_oh.shape[1]


# ============================================================
# TestTrainer
# ============================================================

class TestTrainerWithMonk:
    """使用 Monk 数据测试 MLTrainer。"""

    def test_cv_evaluate_no_hyperparam(self, monk_data):
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        results = trainer.cv_evaluate(X_oh, y, "svm", param_grid=None, verbose=0)

        assert "mean_score" in results
        assert len(results["fold_scores"]) == 3
        assert results["best_params"] is not None

    def test_cv_evaluate_with_param_grid(self, monk_data):
        """关键：测试 param_grid 是展开后的 list[dict]（修复的 bug）。"""
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")

        hp = HyperparamConfig()
        param_grid = get_param_grid("svm", hp.__dict__)
        assert isinstance(param_grid, list)
        assert isinstance(param_grid[0], dict)

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        results = trainer.cv_evaluate(X_oh, y, "svm", param_grid=param_grid, verbose=0)

        assert "mean_score" in results
        assert len(results["fold_scores"]) == 3
        assert isinstance(results["best_params"], dict)

    def test_evaluate_all_models_with_raw_config(self, monk_data):
        """回归测试：从 HyperparamConfig.__dict__ 直接传参（原始 bug）。"""
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")
        hp = HyperparamConfig()

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        # 这就是之前报 TypeError 的调用方式
        results = trainer.evaluate_all_models(
            X_oh, y,
            model_names=["svm"],
            param_grids=hp.__dict__,
            scoring="roc_auc",
            verbose=0,
        )
        assert "svm" in results
        assert results["svm"]["mean_score"] >= 0

    def test_cv_evaluate_with_preprocessing(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")
        pre_cfg = PreprocessConfig(enabled=True, emb_n_components=10)

        trainer = MLTrainer(cv_cfg, out_cfg, preprocess_cfg=pre_cfg, random_state=RANDOM_STATE)
        results = trainer.cv_evaluate(X, y, "svm", param_grid=None, verbose=0)

        assert "mean_score" in results

    def test_oof_predictions_complete(self, monk_data):
        """验证 OOF 预测覆盖所有样本。"""
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        results = trainer.cv_evaluate(X_oh, y, "svm", verbose=0)

        assert len(results["oof_predictions"]) == len(y)
        assert len(results["oof_probabilities"]) == len(y)

    def test_oof_not_using_wrong_fold_model(self, monk_data):
        """OOF 预测应使用当前 fold 的模型，而非之前 fold 的。"""
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=5)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_trainer")

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        results = trainer.cv_evaluate(X_oh, y, "svm", verbose=0)

        # OOF 概率不应该是全 0 或全 1
        oof_proba = np.array(results["oof_probabilities"])
        assert not np.all(oof_proba == oof_proba[0])


# ============================================================
# TestEvaluator
# ============================================================

class TestEvaluatorWithMonk:
    """使用 Monk 数据测试 MLEvaluator。"""

    def test_evaluate(self, monk_data):
        _, X_oh, y, _ = monk_data
        model = create_model("svm", random_state=RANDOM_STATE)
        model.fit(X_oh, y)
        y_pred = model.predict(X_oh)
        y_proba = model.predict_proba(X_oh)[:, 1]

        evaluator = MLEvaluator()
        results = evaluator.evaluate(y, y_pred, y_proba)

        assert "accuracy" in results
        assert "auc_roc" in results
        assert "f1" in results
        assert "confusion_matrix" in results

    def test_compare_models(self, monk_data):
        _, X_oh, y, _ = monk_data
        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_eval")
        hp = HyperparamConfig()

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        all_results = trainer.evaluate_all_models(
            X_oh, y,
            model_names=["svm", "logistic_regression"],
            param_grids=hp.__dict__,
            verbose=0,
        )

        evaluator = MLEvaluator()
        comparisons = evaluator.compare_models(all_results)
        assert len(comparisons) == 2
        assert comparisons[0]["cv_score"] >= comparisons[1]["cv_score"]


# ============================================================
# TestAblator
# ============================================================

class TestAblatorWithMonk:
    """使用 Monk 数据测试消融实验。"""

    def test_ablation_run(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)

        cv_cfg = CVConfig(n_folds=3)
        hp = HyperparamConfig()
        ablation_cfg = AblationConfig(
            modules=["genome_only", "all"],
            save_dir="/tmp/test_ml_ablation",
        )
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_ablation")

        ablator = AblationStudy(
            ablation_cfg, cv_cfg, hp, out_cfg,
            random_state=RANDOM_STATE,
        )
        ablator.run(
            loader,
            model_names=["svm"],
            scoring="roc_auc",
            verbose=0,
        )

        assert "genome_only" in ablator.results
        assert "all" in ablator.results
        assert ablator.best_overall is not None

    def test_ablation_save(self, data_cfg, tmp_path):
        loader = MultiOmicsDataLoader(data_cfg)

        cv_cfg = CVConfig(n_folds=3)
        hp = HyperparamConfig()
        ablation_cfg = AblationConfig(
            modules=["genome_only"],
            save_dir=str(tmp_path / "ablation"),
        )
        out_cfg = OutputConfig(save_dir=str(tmp_path / "ablation"))

        ablator = AblationStudy(
            ablation_cfg, cv_cfg, hp, out_cfg,
            random_state=RANDOM_STATE,
        )
        ablator.run(loader, model_names=["svm"], verbose=0)
        ablator.save_results()

        assert (tmp_path / "ablation" / "ablation_summary.csv").exists()
        assert (tmp_path / "ablation" / "best_config.json").exists()

    def test_ablation_with_raw_param_config(self, data_cfg):
        """消融实验也使用 HyperparamConfig.__dict__，测试不会 TypeError。"""
        loader = MultiOmicsDataLoader(data_cfg)

        cv_cfg = CVConfig(n_folds=3)
        hp = HyperparamConfig()
        ablation_cfg = AblationConfig(
            modules=["genome_only"],
            save_dir="/tmp/test_ml_ablation2",
        )
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_ablation2")

        ablator = AblationStudy(
            ablation_cfg, cv_cfg, hp, out_cfg,
            random_state=RANDOM_STATE,
        )
        # 这应该不会报 TypeError
        ablator.run(
            loader,
            model_names=["svm"],
            scoring="roc_auc",
            verbose=0,
        )
        assert "genome_only" in ablator.results


# ============================================================
# TestConfig
# ============================================================

class TestConfigWithMonk:
    """测试配置加载。"""

    def test_load_config(self, tmp_path):
        config_yaml = tmp_path / "test_config.yaml"
        config_yaml.write_text("""
data:
  emb_dir: "/tmp/emb"
  metab_file: "/tmp/metab.csv"
  pheno_file: "/tmp/pheno.csv"
  label_file: "/tmp/labels.csv"
preprocess:
  enabled: true
  emb_n_components: 64
cv:
  n_folds: 5
hyperparam:
  svm:
    C: [0.1, 1.0]
    kernel: [linear]
ablation:
  modules:
    - genome_only
    - all
output:
  save_dir: "/tmp/test_output"
random_state: 42
n_jobs: 4
""")
        cfg = load_config(str(config_yaml))

        assert cfg.data.emb_dir == "/tmp/emb"
        assert cfg.preprocess.enabled is True
        assert cfg.preprocess.emb_n_components == 64
        assert cfg.cv.n_folds == 5
        assert cfg.random_state == 42
        assert cfg.n_jobs == 4

    def test_global_config_has_preprocess(self):
        """验证 GlobalConfig 包含 preprocess 字段。"""
        cfg = GlobalConfig()
        assert hasattr(cfg, "preprocess")
        assert isinstance(cfg.preprocess, PreprocessConfig)


# ============================================================
# TestEndToEnd
# ============================================================

class TestEndToEndWithMonk:
    """端到端测试：Monk 风格数据 -> 加载 -> 训练 -> 评估。"""

    def test_full_pipeline_no_preprocess(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_e2e")
        hp = HyperparamConfig()

        trainer = MLTrainer(cv_cfg, out_cfg, random_state=RANDOM_STATE)
        all_results = trainer.evaluate_all_models(
            X, y,
            model_names=["svm", "logistic_regression"],
            param_grids=hp.__dict__,
            scoring="roc_auc",
            verbose=0,
        )

        assert len(all_results) == 2
        for name, res in all_results.items():
            assert res["mean_score"] >= 0
            assert len(res["fold_scores"]) == 3
            assert isinstance(res["best_params"], dict)

    def test_full_pipeline_with_preprocess(self, data_cfg):
        loader = MultiOmicsDataLoader(data_cfg)
        X, y, feature_names, sample_ids = loader.load_all()

        cv_cfg = CVConfig(n_folds=3)
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_e2e")
        pre_cfg = PreprocessConfig(enabled=True, emb_n_components=10)
        hp = HyperparamConfig()

        trainer = MLTrainer(cv_cfg, out_cfg, preprocess_cfg=pre_cfg, random_state=RANDOM_STATE)
        all_results = trainer.evaluate_all_models(
            X, y,
            model_names=["svm"],
            param_grids=hp.__dict__,
            scoring="roc_auc",
            verbose=0,
        )

        assert all_results["svm"]["mean_score"] >= 0

    def test_full_pipeline_ablation(self, data_cfg):
        """完整消融实验端到端。"""
        loader = MultiOmicsDataLoader(data_cfg)

        cv_cfg = CVConfig(n_folds=3)
        hp = HyperparamConfig()
        ablation_cfg = AblationConfig(
            modules=["genome_only", "metab_only", "all"],
            save_dir="/tmp/test_ml_e2e_ablation",
        )
        out_cfg = OutputConfig(save_dir="/tmp/test_ml_e2e_ablation")

        ablator = AblationStudy(
            ablation_cfg, cv_cfg, hp, out_cfg,
            random_state=RANDOM_STATE,
        )
        ablator.run(
            loader,
            model_names=["logistic_regression"],
            scoring="roc_auc",
            verbose=0,
        )

        assert len(ablator.results) == 3
        ablator.save_results()
        ablator.print_summary()
