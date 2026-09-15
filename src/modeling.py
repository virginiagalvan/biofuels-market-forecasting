"""
Modeling pipelines, temporal cross-validation, and model comparison:
DummyRegressor (floor) -> Ridge -> Random Forest -> XGBoost -> LightGBM
-> CatBoost.

Each model has its own preprocessing (they don't all need the same
thing): Ridge needs scaling + one-hot, tree models don't need scaling,
and CatBoost can use `country` as a native categorical (107 levels)
without blowing up dimensionality the way one-hot would for the other
models.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = [
    "population",
    "gdp_per_capita",
    "energy_per_capita",
    "oil_consumption",
    "renewables_share_energy",
    "electricity_generation",
    "biofuel_consumption_lag1",
    "biofuel_share_energy_lag1",
    "gdp_per_capita_lag1",
    "biofuel_cagr_5y",
]
CAT_FEATURES_COMMON = ["continent"]
CAT_FEATURES_CATBOOST = ["continent", "country"]
TARGET = "biofuel_consumption"


def _generic_preprocessor(scale: bool):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ] + ([("scale", StandardScaler())] if scale else []))
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="NA")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES_COMMON),
    ])


def make_pipeline(model_name: str, model=None):
    """
    model_name in {"dummy","ridge","random_forest","xgboost","lightgbm","catboost"}
    If `model` is not passed, uses reasonable default hyperparameters.
    """
    if model_name == "dummy":
        model = model or DummyRegressor(strategy="mean")
        return Pipeline([("preprocessor", _generic_preprocessor(scale=False)), ("model", model)])

    if model_name == "ridge":
        model = model or Ridge(alpha=1.0, random_state=42)
        return Pipeline([("preprocessor", _generic_preprocessor(scale=True)), ("model", model)])

    if model_name == "random_forest":
        model = model or RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        return Pipeline([("preprocessor", _generic_preprocessor(scale=False)), ("model", model)])

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        model = model or XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=-1,
        )
        return Pipeline([("preprocessor", _generic_preprocessor(scale=False)), ("model", model)])

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = model or LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=42, verbosity=-1,
        )
        return Pipeline([("preprocessor", _generic_preprocessor(scale=False)), ("model", model)])

    if model_name == "catboost":
        from catboost import CatBoostRegressor
        model = model or CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            cat_features=CAT_FEATURES_CATBOOST, random_seed=42, verbose=False,
        )
        # CatBoost handles NaN and categoricals natively: we only impute
        # numerics and pass the categoricals (including `country`) without one-hot.
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ("cat", SimpleImputer(strategy="constant", fill_value="NA"), CAT_FEATURES_CATBOOST),
        ])
        return CatBoostFriendlyPipeline(pre, model)

    raise ValueError(f"unknown model: {model_name}")


class CatBoostFriendlyPipeline:
    """
    CatBoost needs to know the indices of the categorical columns after
    the ColumnTransformer, which reorders columns. This wrapper keeps
    the fit/predict interface of a scikit-learn Pipeline.
    """

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        n_num = len(NUM_FEATURES)
        n_cat = len(CAT_FEATURES_CATBOOST)
        self.model.set_params(cat_features=list(range(n_num, n_num + n_cat)))

    def fit(self, X, y):
        Xt = self.preprocessor.fit_transform(X)
        self.model.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self.preprocessor.transform(X)
        return self.model.predict(Xt)

    def get_feature_names(self):
        return NUM_FEATURES + CAT_FEATURES_CATBOOST


def expanding_window_splits(df: pd.DataFrame, year_col="year", min_train_years=8):
    """
    Temporal CV by year (expanding window): the first fold trains on
    the first `min_train_years` years and validates on the following
    year; each subsequent fold adds one more year of training. All rows
    (every country) for a given year stay together on the same side of
    the split, never mixed.
    """
    years = sorted(df[year_col].unique())
    splits = []
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_idx = df.index[df[year_col].isin(train_years)].to_numpy()
        test_idx = df.index[df[year_col] == test_year].to_numpy()
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def cross_val_predict_expanding(pipeline_factory, df, splits):
    """
    Runs the pipeline over each fold in `splits` and returns, per fold,
    the predictions and the test year. `pipeline_factory` is a
    zero-argument function that returns a fresh (unfitted) pipeline.
    """
    fold_results = []
    for train_idx, test_idx in splits:
        train, test = df.loc[train_idx], df.loc[test_idx]
        pipe = pipeline_factory()
        pipe.fit(train[NUM_FEATURES + CAT_FEATURES_CATBOOST], train[TARGET])
        preds = pipe.predict(test[NUM_FEATURES + CAT_FEATURES_CATBOOST])
        fold_results.append({
            "year": test["year"].iloc[0],
            "y_true": test[TARGET].to_numpy(),
            "y_pred": preds,
            "countries": test["country"].to_numpy(),
        })
    return fold_results
