"""Nova validator score formula (matches boltz_wrapper._combine_boltz_scores)."""

from __future__ import annotations

from typing import Mapping


def get_heavy_atom_count(smiles: str) -> int:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("rdkit is required for heavy-atom normalization") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return int(mol.GetNumHeavyAtoms())


def validator_boltz_score(
    metrics: Mapping[str, float],
    smiles: str,
    *,
    boltz_metric: tuple[str, ...] = ("affinity_probability_binary", "affinity_pred_value"),
    combination_strategy: str = "heavy_atom_normalization",
) -> float:
    """Replicate Nova validator final score for a single target."""
    if combination_strategy == "average":
        return float(sum(metrics[m] for m in boltz_metric) / len(boltz_metric))

    if combination_strategy == "heavy_atom_normalization":
        if len(boltz_metric) != 2:
            raise ValueError("heavy_atom_normalization requires exactly two metrics")
        heavy_atoms = get_heavy_atom_count(smiles)
        if heavy_atoms == 0:
            raise ValueError("heavy atom count is 0")
        # Nova order: (boltz_metric[0] - boltz_metric[1]) / heavy_atoms
        return float((metrics[boltz_metric[0]] - metrics[boltz_metric[1]]) / heavy_atoms)

    raise ValueError(f"Unsupported combination_strategy: {combination_strategy}")


def local_boltz_score(metrics: Mapping[str, float]) -> float:
    """Score extraction used by score_boltz2.py / BoltzScorer."""
    priority = (
        "affinity_pred_value",
        "affinity_probability_binary",
        "affinity_pred_value1",
        "affinity_probability_binary1",
    )
    for key in priority:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError("No local Boltz score metric found in metrics")
