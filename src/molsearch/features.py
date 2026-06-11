from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover
    RDKit_AVAILABLE = False


def _smiles_descriptors(smiles: str) -> dict[str, float]:
    atom_like = sum(1 for c in smiles if c.isalpha() and c.isupper())
    return {
        "smiles_len": float(len(smiles)),
        "atom_like_count": float(atom_like),
        "ring_count_est": float(sum(ch.isdigit() for ch in smiles)),
        "branch_count_est": float(smiles.count("(") + smiles.count(")")),
    }


def _string_tokens(value: str, prefix: str) -> list[str]:
    tokens = [f"{prefix}_full={value}"]
    if len(value) >= 3:
        tokens.extend(f"{prefix}_tri={value[i:i+3]}" for i in range(len(value) - 2))
    return tokens


def _row_feature_dict(row: pd.Series) -> dict[str, float | str]:
    smiles = str(row.get("smiles", ""))
    molecule_id = str(row.get("molecule_id", ""))

    def _int_or_default(value, default: int = -1) -> int:
        return int(value) if pd.notna(value) else default

    features: dict[str, float | str] = {
        "rxn": _int_or_default(row.get("rxn")),
        "p1": _int_or_default(row.get("p1")),
        "p2": _int_or_default(row.get("p2")),
        "p3": _int_or_default(row.get("p3")),
        "mol_id": f"mol={molecule_id}",
        "smiles_id": f"smiles={smiles}",
    }

    for token in _string_tokens(molecule_id, "mid"):
        features[token] = 1.0
    for token in _string_tokens(smiles, "sm"):
        features[token] = 1.0

    features.update(_smiles_descriptors(smiles))
    return features


def _rdkit_morgan_features(smiles_list: Iterable[str], n_bits: int = 256) -> sparse.csr_matrix:
    smiles_vals = list(smiles_list)
    if not RDKit_AVAILABLE:
        return sparse.csr_matrix((len(smiles_vals), 0))
    arr = np.zeros((len(smiles_vals), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_vals):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        on_bits = list(fp.GetOnBits())
        arr[i, on_bits] = 1.0
    return sparse.csr_matrix(arr)


@dataclass(slots=True)
class FeatureBuilder:
    n_features: int = 1024
    include_rdkit: bool = True
    hasher: FeatureHasher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.hasher = FeatureHasher(n_features=self.n_features, input_type="dict")

    def transform(self, df: pd.DataFrame):
        rows = [_row_feature_dict(row) for _, row in df.iterrows()]
        hashed = self.hasher.transform(rows)
        matrices = [hashed]

        if self.include_rdkit and RDKit_AVAILABLE:
            matrices.append(_rdkit_morgan_features(df["smiles"].astype(str).tolist()))

        return sparse.hstack(matrices, format="csr")
