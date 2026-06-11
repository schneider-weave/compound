from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .config import AppConfig, load_config
from .db import load_molecules
from .history import (
    filter_seen_candidates,
    load_history,
    save_iteration_results,
    top_results,
    update_origin_results,
)
from .molecule_id import add_parsed_columns
from .scorer import BoltzScorer
from .selector import select_batch


def _canonicalize_molecule_id(raw_id: str, cfg: AppConfig) -> str:
    molecule_id = str(raw_id).strip()
    if molecule_id.startswith("rxn:"):
        return molecule_id

    rxn = cfg.search.rxn
    fixed = cfg.search.fixed_params

    # For 2-parameter reactions, map raw free parameter into p2.
    if rxn in {1, 2, 4} and 1 in fixed:
        return f"rxn:{rxn}:{fixed[1]}:{molecule_id}"

    # For 3-parameter reactions with p1 and p2 fixed, map raw free parameter into p3.
    if rxn in {3, 5} and 1 in fixed and 2 in fixed:
        return f"rxn:{rxn}:{fixed[1]}:{fixed[2]}:{molecule_id}"

    return molecule_id


def _normalize_results_file_ids(path: Path, cfg: AppConfig) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "molecule_id" not in df.columns:
        return
    df["molecule_id"] = df["molecule_id"].astype(str).apply(lambda x: _canonicalize_molecule_id(x, cfg))
    df = df.drop_duplicates(subset=["molecule_id"], keep="last")
    if "final_score" in df.columns:
        df = df.sort_values(by="final_score", ascending=False, na_position="last")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _apply_search_filter(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if "rxn" not in df.columns or df["rxn"].dropna().empty:
        print(
            "Warning: molecule IDs in this dataset do not encode rxn/p-parameters. "
            "Skipping rxn/fixed parameter filtering and using all rows."
        )
        return df.reset_index(drop=True)

    out = df[df["rxn"] == cfg.search.rxn].copy()
    for param_idx, value in cfg.search.fixed_params.items():
        col = f"p{param_idx}"
        if col not in out.columns:
            raise ValueError(f"Configured fixed parameter does not exist in parsed data: {col}")
        if out[col].dropna().empty:
            print(
                f"Warning: column {col} is empty for parsed IDs in this dataset. "
                f"Skipping fixed filter {col}={value}."
            )
            continue
        out = out[out[col] == int(value)]
    return out.reset_index(drop=True)


def _prepare_candidates(cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    molecules = load_molecules(cfg.files.molecules_sqlite)
    parsed = add_parsed_columns(molecules)
    filtered = _apply_search_filter(parsed, cfg)
    filtered = filtered.copy()
    filtered["molecule_id"] = filtered["molecule_id"].astype(str).apply(
        lambda x: _canonicalize_molecule_id(x, cfg)
    )
    reparsed = add_parsed_columns(filtered[["molecule_id", "smiles"]])
    filtered["rxn"] = reparsed["rxn"]
    filtered["p1"] = reparsed["p1"]
    filtered["p2"] = reparsed["p2"]
    filtered["p3"] = reparsed["p3"]

    history_df = load_history(cfg.files.origin_results, cfg.files.my_new_results)
    clean = filter_seen_candidates(
        filtered,
        history_df,
        avoid_molecule_id=cfg.duplicate_filter.avoid_molecule_id,
        avoid_smiles=cfg.duplicate_filter.avoid_smiles,
    )

    clean = clean.drop_duplicates(subset=["molecule_id"], keep="first").reset_index(drop=True)
    return clean, history_df


def list_candidates(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    candidates, _ = _prepare_candidates(cfg)
    return candidates


def _score_batch(selected: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if cfg.scoring.mode.lower() == "command" and not cfg.scoring.target:
        raise ValueError(
            "scoring.target must be configured for command mode (Boltz target input is required)."
        )

    scorer = BoltzScorer(
        mode=cfg.scoring.mode,
        command_template=cfg.scoring.command_template,
        timeout_seconds=cfg.scoring.timeout_seconds,
        mock_score=cfg.scoring.mock_score,
        target=cfg.scoring.target,
    )

    rows = []
    now = datetime.now(timezone.utc).isoformat()

    def _to_int_or_nan(value):
        return int(value) if pd.notna(value) else np.nan

    for _, row in selected.iterrows():
        molecule_id = str(row["molecule_id"])
        smiles = str(row["smiles"])
        score = scorer.score(molecule_id, smiles)

        rows.append(
            {
                "molecule_id": molecule_id,
                "rxn": _to_int_or_nan(row.get("rxn")),
                "p1": _to_int_or_nan(row.get("p1")),
                "p2": _to_int_or_nan(row.get("p2")),
                "p3": _to_int_or_nan(row.get("p3")),
                "smiles": smiles,
                "score": score,
                "final_score": score,
                "pred_score": row.get("pred_score", np.nan),
                "uncertainty": row.get("uncertainty", np.nan),
                "acquisition": row.get("acquisition", np.nan),
                "source": "my_run",
                "iteration": cfg.search.iteration,
                "created_at": now,
            }
        )

    scored_df = pd.DataFrame(rows)
    return scored_df


def _print_summary(
    cfg: AppConfig,
    iteration: int,
    candidate_pool_size: int,
    selected_count: int,
    scored_df: pd.DataFrame,
    origin_df: pd.DataFrame,
) -> None:
    failed = int(scored_df["final_score"].isna().sum()) if not scored_df.empty else 0

    iter_best = scored_df.sort_values(by="final_score", ascending=False, na_position="last").head(1)
    overall_best = origin_df.sort_values(by="final_score", ascending=False, na_position="last").head(1)

    fixed = ", ".join([f"p{k}={v}" for k, v in cfg.search.fixed_params.items()]) or "none"

    print(f"Iteration: {iteration}")
    print(f"Reaction type: rxn{cfg.search.rxn}")
    print(f"Fixed params: {fixed}")
    print(f"Candidate pool size: {candidate_pool_size}")
    print(f"Selected count: {selected_count}")
    print(f"Scored count: {len(scored_df)}")
    print(f"Failed count: {failed}")

    if not iter_best.empty and pd.notna(iter_best.iloc[0]["final_score"]):
        print(
            "Best this iteration: "
            f"{iter_best.iloc[0]['molecule_id']} score={iter_best.iloc[0]['final_score']}"
        )

    if not overall_best.empty and pd.notna(overall_best.iloc[0]["final_score"]):
        print(
            "Best overall: "
            f"{overall_best.iloc[0]['molecule_id']} score={overall_best.iloc[0]['final_score']}"
        )


def _run_single_iteration(cfg: AppConfig, iteration: int, dry_run: bool = False) -> pd.DataFrame:
    cfg.search.iteration = iteration

    candidates, history_df = _prepare_candidates(cfg)
    selected = select_batch(candidates, history_df, cfg)

    if dry_run:
        print(f"Dry run selected: {len(selected)}")
        for molecule_id in selected["molecule_id"].tolist():
            print(molecule_id)
        return selected

    scored_df = _score_batch(selected, cfg)

    save_iteration_results(
        scored_df,
        cfg.files.my_new_results,
        sort_by=cfg.results.sort_by,
        sort_order=cfg.results.sort_order,
    )
    _normalize_results_file_ids(cfg.files.my_new_results, cfg)
    origin_df = update_origin_results(
        scored_df,
        cfg.files.origin_results,
        sort_by=cfg.results.sort_by,
        sort_order=cfg.results.sort_order,
    )
    _normalize_results_file_ids(cfg.files.origin_results, cfg)
    origin_df = load_history(cfg.files.origin_results, cfg.files.my_new_results)

    _print_summary(
        cfg,
        iteration=iteration,
        candidate_pool_size=len(candidates),
        selected_count=len(selected),
        scored_df=scored_df,
        origin_df=origin_df,
    )

    return scored_df


def run_iteration(config_path: str, dry_run: bool = False) -> pd.DataFrame:
    cfg = load_config(config_path)
    if dry_run or not cfg.run_control.enabled:
        return _run_single_iteration(cfg, iteration=cfg.search.iteration, dry_run=dry_run)

    start_iter = cfg.search.iteration
    patience = max(cfg.run_control.patience, 1)
    max_iterations = max(cfg.run_control.max_iterations, 1)
    no_improve_streak = 0
    last_avg = -np.inf
    last_best = -np.inf
    last_scored = pd.DataFrame()

    for i in range(max_iterations):
        iteration = start_iter + i
        scored_df = _run_single_iteration(cfg, iteration=iteration, dry_run=False)
        last_scored = scored_df

        valid_scores = pd.to_numeric(scored_df.get("final_score"), errors="coerce").dropna()
        iter_avg = float(valid_scores.mean()) if not valid_scores.empty else -np.inf
        iter_best = float(valid_scores.max()) if not valid_scores.empty else -np.inf

        avg_improved = (iter_avg - last_avg) > cfg.run_control.min_avg_improvement
        best_improved = (iter_best - last_best) > cfg.run_control.min_best_improvement

        if avg_improved:
            last_avg = iter_avg
        if best_improved:
            last_best = iter_best

        if not avg_improved and not best_improved:
            no_improve_streak += 1
        else:
            no_improve_streak = 0

        print(
            "Convergence status: "
            f"avg_improved={avg_improved}, best_improved={best_improved}, "
            f"no_improve_streak={no_improve_streak}/{patience}"
        )

        if no_improve_streak >= patience:
            print(
                f"Early stop triggered after {patience} stagnant iterations "
                "(no avg-score improvement and no new top molecule)."
            )
            break

        candidates_left = len(list_candidates(config_path))
        if candidates_left == 0:
            print("Stopping because candidate pool is exhausted.")
            break

    return last_scored


def show_best(config_path: str, top_n: int) -> pd.DataFrame:
    cfg = load_config(config_path)
    return top_results(cfg.files.origin_results, top_n)
