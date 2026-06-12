#!/usr/bin/env python3
"""Score a single molecule with Nova-validator-aligned Boltz scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molsearch.scorer import BoltzScorer  # noqa: E402
from molsearch.validator_score import (  # noqa: E402
    get_heavy_atom_count,
    raw_affinity_pred_value,
    validator_boltz_score,
)

DEFAULT_TARGET = {
    "name": "Q4QQW4",
    "sequence": (
        "MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKMA"
    ),
}
DEFAULT_MOLECULE_ID = "rxn:1:60111:2212"
DEFAULT_SMILES = "CC1(C(=O)c2cn(I)nn2)COCC1N"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default=DEFAULT_SMILES)
    parser.add_argument("--molecule-id", default=DEFAULT_MOLECULE_ID)
    parser.add_argument("--target-name", default=DEFAULT_TARGET["name"])
    parser.add_argument("--target-sequence", default=DEFAULT_TARGET["sequence"])
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock scorer")
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Skip inference; compute validator score from saved Boltz affinity JSON",
    )
    args = parser.parse_args()

    print("=== Nova validator molecule score ===")
    print(f"molecule_id: {args.molecule_id}")
    print(f"smiles:      {args.smiles}")
    print(f"target:      {args.target_name}")
    print()

    try:
        print(f"heavy_atoms: {get_heavy_atom_count(args.smiles)}")
    except Exception as exc:
        print(f"heavy_atoms: unavailable ({exc})")

    if args.metrics_json:
        metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        score = validator_boltz_score(metrics, args.smiles)
        print()
        print(f"affinity_probability_binary: {metrics.get('affinity_probability_binary')}")
        print(f"affinity_pred_value:         {metrics.get('affinity_pred_value')}")
        print(f"raw affinity_pred_value:     {raw_affinity_pred_value(metrics):.6f}")
        print(f"validator score (Nova):      {score:.6f}")
        return 0

    target = {"name": args.target_name, "sequence": args.target_sequence}
    cmd = [
        sys.executable,
        str(ROOT / "score_boltz2.py"),
        "--smiles",
        args.smiles,
        "--molecule-id",
        args.molecule_id,
        "--target-json",
        json.dumps(target),
    ]
    if args.mock:
        cmd.append("--mock")
    else:
        cmd.append("--strict")

    print()
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print("\nBoltz run failed. On GPU hosts, run scripts/setup-boltz-gpu.sh first.")
        return proc.returncode

    score = BoltzScorer._extract_score(proc.stdout)
    print(f"\nvalidator score (Nova): {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
