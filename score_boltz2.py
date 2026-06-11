from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import yaml


def _ensure_boltz_import_deps() -> None:
    """Nova's Boltz fork imports bittensor only for logging; stub it if missing."""
    try:
        import bittensor  # type: ignore  # noqa: F401
    except ImportError:
        bt = types.ModuleType("bittensor")

        class _BtLogging:
            @staticmethod
            def error(message: str) -> None:
                print(message, file=sys.stderr)

        bt.logging = _BtLogging()
        sys.modules["bittensor"] = bt


def _mock_score(molecule_id: str, smiles: str, target_name: str) -> float:
    digest = hashlib.sha256(f"{molecule_id}|{smiles}|{target_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _load_target(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_json:
        parsed = json.loads(args.target_json)
        if not isinstance(parsed, dict):
            raise ValueError("--target-json must decode to an object")
        return parsed
    return {
        "name": args.target_name,
        "sequence": args.target_sequence,
    }


def _write_boltz_input(target_name: str, target_sequence: str, smiles: str, input_dir: Path) -> Path:
    input_path = input_dir / f"{target_name}.yaml"
    payload = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": target_sequence}},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    input_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return input_path


def _extract_score_from_output_dir(out_dir: Path) -> float:
    priority = [
        "affinity_pred_value",
        "affinity_probability_binary",
        "affinity_pred_value1",
        "affinity_probability_binary1",
    ]
    json_files = list(out_dir.rglob("*.json"))
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            for key in priority:
                if key in data and isinstance(data[key], (int, float)):
                    return float(data[key])
    raise RuntimeError(f"Could not extract affinity score from Boltz outputs in {out_dir}")


def _run_boltz_predict(input_dir: Path, output_dir: Path, cache_dir: str) -> None:
    _ensure_boltz_import_deps()
    try:
        from boltz.main import predict  # type: ignore
    except Exception:
        local_boltz_src = (
            Path(__file__).resolve().parent / "third_party" / "nova" / "external_tools" / "boltz" / "src"
        )
        if local_boltz_src.exists():
            sys.path.insert(0, str(local_boltz_src))
        from boltz.main import predict  # type: ignore

    accelerator = "cpu"
    try:
        import torch  # type: ignore

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        accelerator = "cpu"

    if accelerator == "cpu":
        try:
            import cuequivariance_ops_torch  # type: ignore  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "cuequivariance_ops_torch is unavailable in CPU mode; "
                "cannot run full Boltz inference."
            ) from exc

    try:
        predict(
            data=str(input_dir),
            out_dir=str(output_dir),
            cache=cache_dir,
            override=True,
            num_workers=0,
            use_msa_server=True,
            accelerator=accelerator,
            devices=1,
            no_kernels=(accelerator == "cpu"),
        )
    except Exception as exc:
        raise RuntimeError(f"boltz predict failed in python API: {exc}") from exc


def main() -> int:
    _ensure_boltz_import_deps()
    parser = argparse.ArgumentParser(description="Boltz2 single-molecule scorer.")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--target-sequence", default="")
    parser.add_argument("--target-json", default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail without emitting a score when Boltz inference fails.",
    )
    args = parser.parse_args()

    target = _load_target(args)
    target_name = str(target.get("name", args.target_name or "target"))
    target_sequence = str(target.get("sequence", args.target_sequence or ""))

    if args.mock or not target_sequence:
        score = _mock_score(args.molecule_id, args.smiles, target_name)
        print(f"score: {score:.6f}")
        return 0

    try:
        with tempfile.TemporaryDirectory(prefix="molsearch_boltz_") as tmp:
            tmpdir = Path(tmp)
            input_dir = tmpdir / "inputs"
            output_dir = tmpdir / "outputs"
            cache_dir = os.environ.get("BOLTZ_CACHE", "~/.boltz")
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            _write_boltz_input(target_name, target_sequence, args.smiles, input_dir)
            _run_boltz_predict(input_dir, output_dir, cache_dir)
            score = _extract_score_from_output_dir(output_dir)
            if not math.isfinite(score):
                raise RuntimeError("Boltz score is not finite")
            print(f"score: {score:.6f}")
            return 0
    except Exception as exc:
        print(f"Boltz scoring error: {exc}", file=sys.stderr)
        if args.strict:
            return 2
        fallback = _mock_score(args.molecule_id, args.smiles, target_name)
        print("Falling back to deterministic mock score.", file=sys.stderr)
        print(f"score: {fallback:.6f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
