from types import SimpleNamespace

import pandas as pd

from molsearch.pipeline import _build_role_based_candidates


def test_role_based_candidates_for_single_free_param() -> None:
    molecules = pd.DataFrame(
        [
            {"molecule_id": "10", "smiles": "CCO", "role_mask": 8},
            {"molecule_id": "11", "smiles": "CCC", "role_mask": 12},
            {"molecule_id": "12", "smiles": "CCN", "role_mask": 4},
        ]
    )
    cfg = SimpleNamespace(search=SimpleNamespace(rxn=2, fixed_params={1: 125845}))
    reaction_roles = {2: (4, 8)}

    out = _build_role_based_candidates(molecules, cfg, reaction_roles)
    assert out is not None
    assert out["molecule_id"].tolist() == ["rxn:2:125845:10", "rxn:2:125845:11"]


def test_role_based_candidates_for_two_free_params() -> None:
    molecules = pd.DataFrame(
        [
            {"molecule_id": "20", "smiles": "NCC", "role_mask": 32},
            {"molecule_id": "21", "smiles": "NCO", "role_mask": 32},
            {"molecule_id": "30", "smiles": "OCC", "role_mask": 64},
        ],
    )
    cfg = SimpleNamespace(search=SimpleNamespace(rxn=3, fixed_params={1: 166916}))
    reaction_roles = {3: (16, 32, 64)}

    out = _build_role_based_candidates(molecules, cfg, reaction_roles)
    assert out is not None
    assert len(out) == 2
    assert set(out["molecule_id"]) == {"rxn:3:166916:20:30", "rxn:3:166916:21:30"}
    assert all("." in smi for smi in out["smiles"])


def test_role_based_candidates_rejects_more_than_two_free_params() -> None:
    molecules = pd.DataFrame([{"molecule_id": "10", "smiles": "CCO", "role_mask": 1}])
    cfg = SimpleNamespace(search=SimpleNamespace(rxn=3, fixed_params={}))
    reaction_roles = {3: (16, 32, 64)}

    out = _build_role_based_candidates(molecules, cfg, reaction_roles)
    assert out is None
