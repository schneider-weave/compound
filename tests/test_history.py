import pandas as pd

from molsearch.history import filter_seen_candidates


def test_filter_seen_molecule_ids_and_smiles() -> None:
    candidates = pd.DataFrame(
        [
            {"molecule_id": "rxn:2:1:1", "smiles": "CCO"},
            {"molecule_id": "rxn:2:1:2", "smiles": "CCC"},
            {"molecule_id": "rxn:2:1:3", "smiles": "CCC"},
            {"molecule_id": "rxn:2:1:4", "smiles": "CCN"},
        ]
    )
    history = pd.DataFrame(
        [
            {"molecule_id": "rxn:2:1:1", "smiles": "CCO"},
            {"molecule_id": "rxn:2:99:99", "smiles": "CCN"},
        ]
    )

    filtered = filter_seen_candidates(
        candidates,
        history,
        avoid_molecule_id=True,
        avoid_smiles=True,
    )

    assert "rxn:2:1:1" not in set(filtered["molecule_id"])
    assert "CCN" not in set(filtered["smiles"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["molecule_id"] == "rxn:2:1:2"
