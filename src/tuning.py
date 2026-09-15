"""
Hyperparameter tuning over the temporal cross-validation (expanding
window), not over a single split.

Two levels, on purpose, to show the progression:
- random_search: random sampling of the search space (equivalent to
  RandomizedSearchCV, but compatible with our custom temporal folds
  and with the CatBoost wrapper).
- optuna_search: Bayesian optimization (TPE) over the winning model,
  which learns from previous trials instead of sampling blindly, and
  can prune unpromising trials before they finish.

With a small dataset (~2600 rows) an exhaustive search doesn't make
sense: better to use a bounded search space and say so explicitly than
to "find" a configuration that is actually fitting the noise of the
validation set.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

from .modeling import NUM_FEATURES, CAT_FEATURES_CATBOOST, TARGET, make_pipeline


def _score_params(model_name, params, df, splits):
    maes = []
    for train_idx, test_idx in splits:
        train, test = df.loc[train_idx], df.loc[test_idx]
        pipe = make_pipeline(model_name, model=_build_model(model_name, params))
        pipe.fit(train[NUM_FEATURES + CAT_FEATURES_CATBOOST], train[TARGET])
        preds = pipe.predict(test[NUM_FEATURES + CAT_FEATURES_CATBOOST])
        maes.append(mean_absolute_error(test[TARGET], preds))
    return float(np.mean(maes))


def _build_model(model_name, params):
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(random_state=42, **params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, n_jobs=-1, **params)
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, verbosity=-1, **params)
    if model_name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            cat_features=CAT_FEATURES_CATBOOST, random_seed=42, verbose=False, **params
        )
    raise ValueError(model_name)


def random_search(model_name, param_distributions, df, splits, n_iter=20, random_state=42):
    """
    param_distributions: dict {param: list_of_possible_values}.
    Returns (best_params, history) with the average MAE of each
    combination tried, so the process can be shown in the notebook.
    """
    rng = np.random.default_rng(random_state)
    keys = list(param_distributions.keys())
    history = []
    for _ in range(n_iter):
        params = {k: rng.choice(param_distributions[k]).item()
                  if hasattr(rng.choice(param_distributions[k]), "item")
                  else rng.choice(param_distributions[k])
                  for k in keys}
        mae = _score_params(model_name, params, df, splits)
        history.append({"params": params, "MAE": mae})
    history.sort(key=lambda h: h["MAE"])
    return history[0]["params"], history


def optuna_search(model_name, param_space_fn, df, splits, n_trials=30, random_state=42):
    """
    param_space_fn(trial) -> dict of hyperparameters, using the Optuna
    API (trial.suggest_int/suggest_float/...). Returns (best_params, study).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = param_space_fn(trial)
        return _score_params(model_name, params, df, splits)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study
