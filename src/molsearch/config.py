from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SearchConfig:
    rxn: int
    fixed_params: dict[int, int]
    batch_size: int
    iteration: int


@dataclass(slots=True)
class FilesConfig:
    molecules_sqlite: Path
    origin_results: Path
    my_new_results: Path


@dataclass(slots=True)
class DuplicateFilterConfig:
    avoid_molecule_id: bool
    avoid_smiles: bool


@dataclass(slots=True)
class SelectionConfig:
    method: str
    beta: float
    exploit_count: int
    ucb_count: int
    explore_count: int
    min_training_rows: int
    random_seed: int
    model: str = "extra_trees"


@dataclass(slots=True)
class ScoringConfig:
    mode: str
    command_template: str
    timeout_seconds: int
    mock_score: bool
    target: dict[str, Any]


@dataclass(slots=True)
class ResultsConfig:
    sort_by: str
    sort_order: str


@dataclass(slots=True)
class RunControlConfig:
    enabled: bool
    patience: int
    max_iterations: int
    min_avg_improvement: float
    min_best_improvement: float


@dataclass(slots=True)
class AppConfig:
    search: SearchConfig
    files: FilesConfig
    duplicate_filter: DuplicateFilterConfig
    selection: SelectionConfig
    scoring: ScoringConfig
    results: ResultsConfig
    run_control: RunControlConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config key: {key}")
    return mapping[key]


def _to_int_dict(raw: dict[Any, Any]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for key, value in raw.items():
        parsed[int(key)] = int(value)
    return parsed


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    search_raw = _require(cfg, "search")
    files_raw = _require(cfg, "files")
    dup_raw = _require(cfg, "duplicate_filter")
    selection_raw = _require(cfg, "selection")
    scoring_raw = _require(cfg, "scoring")
    results_raw = _require(cfg, "results")
    run_control_raw = cfg.get("run_control", {})

    root = path.parent
    rxn_value = int(_require(search_raw, "rxn"))

    def _resolve_file_path(raw_path: str) -> Path:
        rendered = str(raw_path).format(rxn=rxn_value)
        return (root / rendered).resolve()

    files = FilesConfig(
        molecules_sqlite=_resolve_file_path(files_raw["molecules_sqlite"]),
        origin_results=_resolve_file_path(files_raw["origin_results"]),
        my_new_results=_resolve_file_path(files_raw["my_new_results"]),
    )

    return AppConfig(
        search=SearchConfig(
            rxn=rxn_value,
            fixed_params=_to_int_dict(search_raw.get("fixed_params", {})),
            batch_size=int(_require(search_raw, "batch_size")),
            iteration=int(search_raw.get("iteration", 1)),
        ),
        files=files,
        duplicate_filter=DuplicateFilterConfig(
            avoid_molecule_id=bool(dup_raw.get("avoid_molecule_id", True)),
            avoid_smiles=bool(dup_raw.get("avoid_smiles", True)),
        ),
        selection=SelectionConfig(
            method=str(selection_raw.get("method", "active_search")),
            beta=float(selection_raw.get("beta", 1.2)),
            exploit_count=int(selection_raw.get("exploit_count", 40)),
            ucb_count=int(selection_raw.get("ucb_count", 15)),
            explore_count=int(selection_raw.get("explore_count", 5)),
            min_training_rows=int(selection_raw.get("min_training_rows", 30)),
            random_seed=int(selection_raw.get("random_seed", 42)),
            model=str(selection_raw.get("model", "extra_trees")),
        ),
        scoring=ScoringConfig(
            mode=str(scoring_raw.get("mode", "command")),
            command_template=str(scoring_raw.get("command_template", "")),
            timeout_seconds=int(scoring_raw.get("timeout_seconds", 3600)),
            mock_score=bool(scoring_raw.get("mock_score", False)),
            target=dict(scoring_raw.get("target", {})),
        ),
        results=ResultsConfig(
            sort_by=str(results_raw.get("sort_by", "score")),
            sort_order=str(results_raw.get("sort_order", "desc")),
        ),
        run_control=RunControlConfig(
            enabled=bool(run_control_raw.get("enabled", True)),
            patience=int(run_control_raw.get("patience", 5)),
            max_iterations=int(run_control_raw.get("max_iterations", 100)),
            min_avg_improvement=float(run_control_raw.get("min_avg_improvement", 0.0)),
            min_best_improvement=float(run_control_raw.get("min_best_improvement", 0.0)),
        ),
    )
