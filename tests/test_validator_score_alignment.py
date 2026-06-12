"""Validator vs local Boltz score alignment tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from molsearch.scorer import BoltzScorer
from molsearch.validator_score import get_heavy_atom_count, local_boltz_score, validator_boltz_score

ROOT = Path(__file__).resolve().parents[1]
SCORE_SCRIPT = ROOT / "score_boltz2.py"

TARGET_NAME = "Q4QQW4"
TARGET_SEQUENCE = (
    "MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKMA"
)
MOLECULE_ID = "rxn:3:61930:357:102046"
SMILES = "COc1ccsc1CNC(=O)[C@@H](CCSC)n1cc(-c2ccc(C)s2)nn1"


def test_user_molecule_heavy_atom_count() -> None:
    pytest.importorskip("rdkit")
    assert get_heavy_atom_count(SMILES) == 27


def test_validator_and_local_scores_differ_for_same_raw_metrics() -> None:
    pytest.importorskip("rdkit")
    # Typical Boltz scales: prob in [0,1], pred_value is log10(IC50) often negative.
    metrics = {
        "affinity_probability_binary": 0.72,
        "affinity_pred_value": -2.83,
    }
    local = local_boltz_score(metrics)
    validator = validator_boltz_score(metrics, SMILES)
    assert local == pytest.approx(-2.83)
    assert validator == pytest.approx((0.72 - (-2.83)) / 27)
    assert validator == pytest.approx(0.1315, rel=1e-3)
    assert local != pytest.approx(validator)


def test_validator_formula_matches_nova_wrapper_example() -> None:
    pytest.importorskip("rdkit")
    metrics = {
        "affinity_probability_binary": 0.72,
        "affinity_pred_value": -2.83,
    }
    expected = (0.72 - (-2.83)) / get_heavy_atom_count(SMILES)
    assert validator_boltz_score(metrics, SMILES) == pytest.approx(expected)
    assert validator_boltz_score(metrics, SMILES) == pytest.approx(0.1315, rel=1e-3)


def test_q4qqw4_validator_scores_never_exceed_0_2() -> None:
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "nova_results_Molecules_RXN2.csv")
    scores = df.loc[df["target_protein"] == "Q4QQW4", "final_score"]
    assert float(scores.max()) < 0.2
    assert float(scores.max()) == pytest.approx(0.167271, rel=1e-3)


def test_mock_cli_score_for_user_molecule() -> None:
    target_json = json.dumps({"name": TARGET_NAME, "sequence": TARGET_SEQUENCE})
    proc = subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            "--mock",
            "--smiles",
            SMILES,
            "--molecule-id",
            MOLECULE_ID,
            "--target-json",
            target_json,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    score = BoltzScorer._extract_score(proc.stdout)
    assert score == pytest.approx(0.246567, rel=1e-5)


@pytest.mark.integration
def test_real_boltz_cli_for_user_molecule() -> None:
    """Run on a GPU host with Boltz deps installed. Skipped when inference is unavailable."""
    target_json = json.dumps({"name": TARGET_NAME, "sequence": TARGET_SEQUENCE})
    proc = subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            "--strict",
            "--smiles",
            SMILES,
            "--molecule-id",
            MOLECULE_ID,
            "--target-json",
            target_json,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        timeout=7200,
        env={**__import__("os").environ, "BOLTZ_CACHE": "data/boltz-cache"},
    )
    if proc.returncode != 0:
        pytest.skip(f"Boltz inference unavailable: {proc.stderr.strip()}")

    local = BoltzScorer._extract_score(proc.stdout)
    assert math.isfinite(local)
