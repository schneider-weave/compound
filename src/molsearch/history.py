from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_HISTORY_COLUMNS = ["molecule_id", "smiles", "score", "final_score", "created_at"]
MY_RESULTS_COLUMNS = ["molecule_id", "final_score"]


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_HISTORY_COLUMNS)


def load_results_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return _empty_results()
    df = pd.read_csv(p)
    if "molecule_id" not in df.columns:
        raise ValueError(f"Results file is missing required column `molecule_id`: {p}")

    if "score" not in df.columns and "final_score" in df.columns:
        df["score"] = pd.to_numeric(df["final_score"], errors="coerce")
    if "final_score" not in df.columns and "score" in df.columns:
        df["final_score"] = pd.to_numeric(df["score"], errors="coerce")

    for col in CANONICAL_HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CANONICAL_HISTORY_COLUMNS]


def filter_valid_history(
    history_df: pd.DataFrame,
    require_smiles: bool = True,
    max_score: float | None = None,
) -> pd.DataFrame:
    if history_df.empty:
        return history_df.copy()

    out = history_df.copy()
    scores = pd.to_numeric(out.get("final_score", out.get("score")), errors="coerce")
    out["final_score"] = scores
    out["score"] = scores
    out = out[scores.notna()]

    if require_smiles and "smiles" in out.columns:
        smiles = out["smiles"].astype(str).str.strip()
        out = out[smiles.notna() & (smiles != "") & (smiles.str.lower() != "nan")]

    if max_score is not None:
        out = out[out["final_score"] <= float(max_score)]

    return out.reset_index(drop=True)


def load_history(origin_results: str | Path, my_new_results: str | Path) -> pd.DataFrame:
    origin_df = load_results_csv(origin_results)
    my_df = load_results_csv(my_new_results)
    merged = my_df.copy() if origin_df.empty else (
        origin_df.copy() if my_df.empty else pd.concat([origin_df, my_df], ignore_index=True)
    )
    if merged.empty:
        return _empty_results()
    merged = merged.drop_duplicates(subset=["molecule_id"], keep="last")
    return merged


def filter_seen_candidates(
    candidates: pd.DataFrame,
    history_df: pd.DataFrame,
    avoid_molecule_id: bool,
    avoid_smiles: bool,
) -> pd.DataFrame:
    df = candidates.copy()

    if avoid_molecule_id and not history_df.empty:
        seen_ids = set(history_df["molecule_id"].dropna().astype(str))
        df = df[~df["molecule_id"].astype(str).isin(seen_ids)]

    if avoid_smiles:
        seen_smiles: set[str] = set()
        if not history_df.empty:
            seen_smiles = set(history_df["smiles"].dropna().astype(str))

        df = df[~df["smiles"].astype(str).isin(seen_smiles)]
        df = df.drop_duplicates(subset=["smiles"], keep="first")

    return df.reset_index(drop=True)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _sort_results(df: pd.DataFrame, sort_by: str, ascending: bool) -> pd.DataFrame:
    if sort_by not in df.columns:
        return df
    return df.sort_values(
        by=sort_by,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def save_iteration_results(
    new_rows: pd.DataFrame,
    my_new_results: str | Path,
    sort_by: str = "score",
    sort_order: str = "desc",
) -> pd.DataFrame:
    path = Path(my_new_results)
    current = load_results_csv(path)
    normalized = new_rows.copy()
    if "final_score" not in normalized.columns:
        normalized["final_score"] = pd.to_numeric(normalized.get("score"), errors="coerce")
    normalized = normalized[MY_RESULTS_COLUMNS]
    normalized = normalized.dropna(subset=["final_score"])

    merged = normalized if current.empty else pd.concat(
        [current[MY_RESULTS_COLUMNS], normalized],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(subset=["molecule_id"], keep="last")
    sorted_df = _sort_results(
        merged,
        sort_by="final_score" if sort_by == "score" else sort_by,
        ascending=(sort_order.lower() == "asc"),
    )
    _atomic_write_csv(sorted_df[MY_RESULTS_COLUMNS], path)
    return sorted_df[MY_RESULTS_COLUMNS]


def update_origin_results(
    new_rows: pd.DataFrame,
    origin_results: str | Path,
    sort_by: str = "score",
    sort_order: str = "desc",
) -> pd.DataFrame:
    path = Path(origin_results)
    current_raw = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=MY_RESULTS_COLUMNS)
    normalized = new_rows.copy()
    if "final_score" not in normalized.columns:
        normalized["final_score"] = pd.to_numeric(normalized.get("score"), errors="coerce")
    normalized = normalized[MY_RESULTS_COLUMNS]
    normalized = normalized.dropna(subset=["final_score"])

    for col in current_raw.columns:
        if col not in normalized.columns:
            normalized[col] = np.nan
    for col in normalized.columns:
        if col not in current_raw.columns:
            current_raw[col] = np.nan

    merged = normalized if current_raw.empty else pd.concat(
        [current_raw[normalized.columns], normalized],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(subset=["molecule_id"], keep="last")
    sorted_df = _sort_results(
        merged,
        sort_by="final_score" if sort_by == "score" else sort_by,
        ascending=(sort_order.lower() == "asc"),
    )
    _atomic_write_csv(sorted_df, path)
    return load_results_csv(path)


def top_results(path: str | Path, top_n: int) -> pd.DataFrame:
    df = load_results_csv(path)
    if df.empty:
        return df
    sort_col = "final_score" if "final_score" in df.columns else "score"
    scored = df.sort_values(by=sort_col, ascending=False, na_position="last")
    return scored.head(top_n).reset_index(drop=True)
