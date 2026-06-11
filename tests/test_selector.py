from types import SimpleNamespace

import numpy as np
import pandas as pd

from molsearch.selector import select_batch


def _make_config(batch_size: int = 60):
    return SimpleNamespace(
        search=SimpleNamespace(batch_size=batch_size),
        selection=SimpleNamespace(
            beta=1.2,
            exploit_count=40,
            ucb_count=15,
            explore_count=5,
            min_training_rows=30,
            random_seed=42,
            model="extra_trees",
        ),
    )


def _candidates(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "molecule_id": f"rxn:2:166916:{100000+i}",
                "smiles": f"C{'C' * (i % 10)}N{i%7}",
                "rxn": 2,
                "p1": 166916,
                "p2": 100000 + i,
                "p3": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _history(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "molecule_id": f"rxn:2:166916:{200000+i}",
                "smiles": f"N{'C' * (i % 6)}O{i%5}",
                "rxn": 2,
                "p1": 166916,
                "p2": 200000 + i,
                "p3": np.nan,
                "score": float(i) / n,
            }
        )
    return pd.DataFrame(rows)


def test_selector_returns_batch_size_when_enough_candidates() -> None:
    selected = select_batch(_candidates(200), _history(100), _make_config(batch_size=60))
    assert len(selected) == 60
    assert selected["molecule_id"].nunique() == 60


def test_selector_returns_fewer_when_candidate_pool_is_smaller() -> None:
    selected = select_batch(_candidates(22), _history(100), _make_config(batch_size=60))
    assert len(selected) == 22
    assert selected["molecule_id"].nunique() == 22
