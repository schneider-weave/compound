from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .features import FeatureBuilder

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    CATBOOST_AVAILABLE = False


@dataclass(slots=True)
class SelectionSettings:
    batch_size: int
    beta: float
    exploit_count: int
    ucb_count: int
    explore_count: int
    min_training_rows: int
    random_seed: int
    model: str = "extra_trees"


def _pick_diverse(candidates: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
    if len(candidates) <= n:
        return candidates.copy()

    df = candidates.copy()
    df = df.drop_duplicates(subset=["smiles"], keep="first")

    sort_cols = [c for c in ["p1", "p2", "p3"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)

    if len(df) <= n:
        return df

    positions = np.linspace(0, len(df) - 1, num=n, dtype=int)
    selected = df.iloc[np.unique(positions)].copy()

    if len(selected) < n:
        remainder = df.drop(index=selected.index)
        fill = remainder.sample(n=n - len(selected), random_state=random_seed)
        selected = pd.concat([selected, fill], ignore_index=True)

    return selected.head(n).reset_index(drop=True)


def _estimate_uncertainty_extra_trees(model: ExtraTreesRegressor, x):
    tree_preds = np.vstack([tree.predict(x) for tree in model.estimators_])
    return tree_preds.std(axis=0)


def _train_model(settings: SelectionSettings, x_train, y_train):
    model_name = settings.model.lower()

    if model_name == "catboost" and CATBOOST_AVAILABLE:
        model = CatBoostRegressor(
            random_seed=settings.random_seed,
            verbose=False,
            depth=8,
            n_estimators=500,
            learning_rate=0.05,
        )
        model.fit(x_train, y_train)
        return model

    model = ExtraTreesRegressor(
        n_estimators=400,
        random_state=settings.random_seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)
    return model


def select_batch(candidates: pd.DataFrame, history_df: pd.DataFrame, config) -> pd.DataFrame:
    settings = SelectionSettings(
        batch_size=int(config.search.batch_size),
        beta=float(config.selection.beta),
        exploit_count=int(config.selection.exploit_count),
        ucb_count=int(config.selection.ucb_count),
        explore_count=int(config.selection.explore_count),
        min_training_rows=int(config.selection.min_training_rows),
        random_seed=int(config.selection.random_seed),
        model=str(getattr(config.selection, "model", "extra_trees")),
    )

    if candidates.empty:
        return candidates.copy()

    candidates = candidates.drop_duplicates(subset=["molecule_id"], keep="first").reset_index(drop=True)

    if len(candidates) <= settings.batch_size:
        out = candidates.copy()
        out["pred_score"] = np.nan
        out["uncertainty"] = np.nan
        out["acquisition"] = np.nan
        return out

    history = history_df.copy()
    history["score"] = pd.to_numeric(history.get("score", np.nan), errors="coerce")
    training = history.dropna(subset=["score", "molecule_id", "smiles"]).copy()

    if len(training) < settings.min_training_rows:
        selected = _pick_diverse(candidates, settings.batch_size, settings.random_seed)
        selected["pred_score"] = np.nan
        selected["uncertainty"] = np.nan
        selected["acquisition"] = np.nan
        return selected

    feature_builder = FeatureBuilder(include_rdkit=True)
    x_train = feature_builder.transform(training)
    y_train = training["score"].astype(float).values
    model = _train_model(settings, x_train, y_train)

    x_cand = feature_builder.transform(candidates)
    pred = model.predict(x_cand)

    if isinstance(model, ExtraTreesRegressor):
        uncertainty = _estimate_uncertainty_extra_trees(model, x_cand)
    else:
        uncertainty = np.full(shape=len(candidates), fill_value=float(np.std(y_train)))

    # Novelty bonus based on parameter rarity within candidate pool.
    param_cols = [c for c in ["p1", "p2", "p3"] if c in candidates.columns]
    novelty_bonus = np.zeros(len(candidates), dtype=float)
    if param_cols:
        for col in param_cols:
            counts = candidates[col].value_counts(dropna=False)
            novelty_bonus += candidates[col].map(lambda x: 1.0 / counts.get(x, 1)).to_numpy()
        novelty_bonus = novelty_bonus / max(len(param_cols), 1)

    acquisition = pred + settings.beta * uncertainty + novelty_bonus

    scored = candidates.copy()
    scored["pred_score"] = pred
    scored["uncertainty"] = uncertainty
    scored["acquisition"] = acquisition

    selected_parts: list[pd.DataFrame] = []

    exploit = scored.nlargest(settings.exploit_count, "pred_score")
    selected_parts.append(exploit)

    remain = scored[~scored["molecule_id"].isin(exploit["molecule_id"])]
    ucb = remain.nlargest(settings.ucb_count, "acquisition")
    selected_parts.append(ucb)

    remain = remain[~remain["molecule_id"].isin(ucb["molecule_id"])]
    explore = _pick_diverse(remain, settings.explore_count, settings.random_seed)
    selected_parts.append(explore)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates(subset=["molecule_id"], keep="first")

    if len(selected) < settings.batch_size:
        remain = scored[~scored["molecule_id"].isin(selected["molecule_id"])]
        needed = settings.batch_size - len(selected)
        fill = _pick_diverse(remain, needed, settings.random_seed)
        selected = pd.concat([selected, fill], ignore_index=True)
        selected = selected.drop_duplicates(subset=["molecule_id"], keep="first")

    return selected.head(settings.batch_size).reset_index(drop=True)
