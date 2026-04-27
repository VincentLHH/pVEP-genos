"""
apps.ml.models
==============
机器学习模型定义。

支持：
- SVM (Support Vector Machine)
- LR (Logistic Regression)
- XGBoost
- MLP (Multi-Layer Perceptron)
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None


# 可用模型注册表
AVAILABLE_MODELS = ["svm", "logistic_regression", "xgboost", "mlp"]


class StandardizableModel:
    """
    支持自动标准化的模型封装。

    训练时自动对特征进行标准化（fit时同步fit scaler），
    预测时同步transform。
    """

    def __init__(self, model, need_scaling: bool = True):
        self.model = model
        self.need_scaling = need_scaling
        self.scaler = StandardScaler() if need_scaling else None

    def fit(self, X, y, **kwargs):
        if self.need_scaling and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        if self.need_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.need_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def score(self, X, y):
        if self.need_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.score(X, y)

    @property
    def feature_importances_(self):
        """返回特征重要性（如模型支持）"""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # LR/SVM: 使用系数绝对值
            return np.abs(self.model.coef_).mean(axis=0)
        return None

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


def create_model(model_name: str, random_state: int = 42, **model_params) -> StandardizableModel:
    """
    工厂函数：创建ML模型。

    Args:
        model_name: 模型名（svm/logistic_regression/xgboost/mlp）
        random_state: 随机种子
        **model_params: 模型特定参数

    Returns:
        StandardizableModel 封装实例
    """
    _SVC_PARAMS = {
        "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability",
        "tol", "cache_size", "class_weight", "verbose", "max_iter",
        "decision_function_shape", "break_ties", "random_state",
    }
    _LR_PARAMS = {
        "C", "penalty", "dual", "tol", "fit_intercept", "intercept_scaling",
        "class_weight", "solver", "max_iter", "multi_class", "verbose",
        "warm_start", "n_jobs", "l1_ratio", "random_state",
    }
    _XGB_PARAMS = {
        "n_estimators", "max_depth", "max_leaves", "learning_rate",
        "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
        "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
        "scale_pos_weight", "objective", "eval_metric",
    }
    _MLP_PARAMS = {
        "hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
        "learning_rate", "learning_rate_init", "power_t", "max_iter",
        "shuffle", "random_state", "tol", "verbose", "warm_start",
        "momentum", "nesterovs_momentum", "early_stopping",
        "validation_fraction", "beta_1", "beta_2", "epsilon",
        "n_iter_no_change", "max_fun",
        "dropout",  # not a native sklearn param; handled below
    }

    if model_name == "svm":
        extra = set(model_params) - _SVC_PARAMS
        if extra:
            import warnings
            warnings.warn(f"SVC ignoring unknown params: {extra}")
        model = SVC(
            probability=True,
            random_state=random_state,
            **{k: v for k, v in model_params.items() if k in _SVC_PARAMS},
        )
        return StandardizableModel(model, need_scaling=True)

    elif model_name == "logistic_regression":
        extra = set(model_params) - _LR_PARAMS
        if extra:
            import warnings
            warnings.warn(f"LogisticRegression ignoring unknown params: {extra}")
        model = LogisticRegression(
            random_state=random_state,
            **{k: v for k, v in model_params.items() if k in _LR_PARAMS},
        )
        return StandardizableModel(model, need_scaling=True)

    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost未安装。请运行: pip install xgboost")
        extra = set(model_params) - _XGB_PARAMS
        if extra:
            import warnings
            warnings.warn(f"XGBoost ignoring unknown params: {extra}")
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            **{k: v for k, v in model_params.items() if k in _XGB_PARAMS},
        }
        model = xgb.XGBClassifier(**params)
        return StandardizableModel(model, need_scaling=False)

    elif model_name == "mlp":
        extra = set(model_params) - _MLP_PARAMS
        if extra:
            import warnings
            warnings.warn(f"MLP ignoring unknown params: {extra}")
        mlp_params = {k: v for k, v in model_params.items() if k in _MLP_PARAMS}

        # dropout 不是 sklearn 原生参数，转换为 alpha 正则化
        if "dropout" in mlp_params:
            dropout = mlp_params.pop("dropout")
            if dropout > 0:
                mlp_params["alpha"] = max(mlp_params.get("alpha", 0.0001), dropout * 0.1)

        model = MLPClassifier(
            random_state=random_state,
            **mlp_params,
        )
        return StandardizableModel(model, need_scaling=True)

    else:
        raise ValueError(f"未知模型: {model_name}. 可选: {AVAILABLE_MODELS}")


def get_param_grid(model_name: str, hyperparam_cfg: Dict) -> List[Dict]:
    """
    从超参配置生成参数网格。

    Args:
        model_name: 模型名
        hyperparam_cfg: 超参搜索空间配置

    Returns:
        参数网格列表（用于GridSearchCV）
    """
    import itertools

    if model_name not in hyperparam_cfg:
        return [{}]

    param_space = hyperparam_cfg[model_name]

    # 转换为 {param: [values]} 格式
    grid = []
    for param, values in param_space.items():
        if isinstance(values, list):
            grid.append((param, values))
        else:
            grid.append((param, [values]))

    if not grid:
        return [{}]

    # 生成所有组合
    keys = [k for k, _ in grid]
    value_lists = [v for _, v in grid]

    param_dicts = []
    for values in itertools.product(*value_lists):
        param_dicts.append(dict(zip(keys, values)))

    return param_dicts
