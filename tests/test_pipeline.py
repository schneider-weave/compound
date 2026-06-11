import pandas as pd

from molsearch.pipeline import _pool_entropy


def test_pool_entropy_handles_mixed_numeric_and_nan_param_values() -> None:
    selected = pd.DataFrame(
        [
            {"p1": 10, "p2": 1.0, "p3": float("nan"), "smiles": "CCO"},
            {"p1": 10, "p2": 1.0, "p3": float("nan"), "smiles": "CCC"},
            {"p1": 10, "p2": 2.0, "p3": 3.0, "smiles": "CCN"},
        ]
    )
    entropy = _pool_entropy(selected)
    assert entropy >= 0.0
