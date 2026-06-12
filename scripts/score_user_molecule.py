#!/usr/bin/env python3
"""Score a single molecule with local Boltz and show validator-equivalent score."""

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
    local_boltz_score,
    validator_boltz_score,
)

DEFAULT_TARGET = {
    "name": "Q4QQW4",
    "sequence": (
        "MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKMA"
    ),
}
DEFAULT_MOLECULE_ID = "rxn:3:61930:357:102046"
DEFAULT_SMILES = "COc1ccsc1CNC(=O)[C@@H](CCSC)n1cc(-c2ccc(C)s2)nn1"


def _load_metrics_from_boltz_output(output_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path in output_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
    return metrics


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

    print("=== Molecule scoring comparison ===")
    print(f"molecule_id: {args.molecule_id}")
    print(f"smiles:      {args.smiles}")
    print(f"target:      {args.target_name}")
    print()

    try:
        heavy_atoms = get_heavy_atom_count(args.smiles)
        print(f"heavy_atoms: {heavy_atoms}")
    except Exception as exc:
        print(f"heavy_atoms: unavailable ({exc})")
        heavy_atoms = None

    if args.metrics_json:
        metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        local = local_boltz_score(metrics)
        validator = validator_boltz_score(metrics, args.smiles)
        print()
        print("From saved Boltz metrics:")
        print(f"  affinity_pred_value:          {metrics.get('affinity_pred_value')}")
        print(f"  affinity_probability_binary:  {metrics.get('affinity_probability_binary')}")
        print(f"  local score (this repo):      {local:.6f}")
        print(f"  validator score (Nova):       {validator:.6f}")
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

    local = BoltzScorer._extract_score(proc.stdout)
    print(f"\nlocal score (this repo): {local:.6f}")

    if args.mock:
        print("mock mode: validator score requires real Boltz affinity JSON metrics")
        return 0

    print(
        "\nTo compute validator score after a real run, pass the affinity JSON:\n"
        "  python scripts/score_user_molecule.py --metrics-json path/to/affinity_*.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
