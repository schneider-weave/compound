"""Nova validator score formula and Boltz affinity metric helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

# From metanova-labs/nova config/boltz_config.yaml
NOVA_BOLTZ_PREDICT_KWARGS: dict[str, object] = {
    "seed": 68,
    "recycling_steps": 3,
    "sampling_steps": 100,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 100,
    "diffusion_samples_affinity": 3,
    "affinity_mw_correction": True,
    "override": False,
    "output_format": "mmcif",
}

# From metanova-labs/nova config/config.yaml boltz2_config
NOVA_BOLTZ_METRIC: tuple[str, ...] = ("affinity_probability_binary", "affinity_pred_value")
NOVA_COMBINATION_STRATEGY = "heavy_atom_normalization"

AFFINITY_METRIC_KEYS = (
    "affinity_probability_binary",
    "affinity_pred_value",
    "affinity_probability_binary1",
    "affinity_pred_value1",
    "affinity_probability_binary2",
    "affinity_pred_value2",
)


def get_heavy_atom_count(smiles: str) -> int:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("rdkit is required for heavy-atom normalization") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return int(mol.GetNumHeavyAtoms())


def extract_affinity_metrics(payload: object) -> dict[str, float]:
    """Collect numeric Boltz affinity fields from a JSON object (possibly nested)."""
    metrics: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in AFFINITY_METRIC_KEYS and isinstance(value, (int, float)):
                metrics[key] = float(value)
            else:
                metrics.update(extract_affinity_metrics(value))
    elif isinstance(payload, list):
        for item in payload:
            metrics.update(extract_affinity_metrics(item))
    return metrics


def extract_affinity_metrics_from_dir(out_dir: Path) -> dict[str, float]:
    """Load affinity metrics from Boltz prediction JSON files."""
    merged: dict[str, float] = {}
    for path in out_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, value in extract_affinity_metrics(data).items():
            merged[key] = value
    return merged


def validator_boltz_score(
    metrics: Mapping[str, float],
    smiles: str,
    *,
    boltz_metric: tuple[str, ...] = NOVA_BOLTZ_METRIC,
    combination_strategy: str = NOVA_COMBINATION_STRATEGY,
) -> float:
    """Replicate Nova validator final_score for a single target."""
    if combination_strategy == "average":
        return float(sum(metrics[m] for m in boltz_metric) / len(boltz_metric))

    if combination_strategy == "heavy_atom_normalization":
        if len(boltz_metric) != 2:
            raise ValueError("heavy_atom_normalization requires exactly two metrics")
        missing = [m for m in boltz_metric if m not in metrics]
        if missing:
            raise KeyError(f"Missing Boltz metrics for validator score: {missing}")
        heavy_atoms = get_heavy_atom_count(smiles)
        if heavy_atoms == 0:
            raise ValueError("heavy atom count is 0")
        # Nova order: (boltz_metric[0] - boltz_metric[1]) / heavy_atoms
        return float((metrics[boltz_metric[0]] - metrics[boltz_metric[1]]) / heavy_atoms)

    raise ValueError(f"Unsupported combination_strategy: {combination_strategy}")


def raw_affinity_pred_value(metrics: Mapping[str, float]) -> float:
    """Raw Boltz affinity_pred_value (log10 IC50), not the validator final score."""
    for key in ("affinity_pred_value", "affinity_pred_value1"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError("affinity_pred_value not found in metrics")
