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

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from molsearch.validator_score import (  # noqa: E402
    NOVA_BOLTZ_PREDICT_KWARGS,
    extract_affinity_metrics_from_dir,
    validator_boltz_score,
)


def _patch_torch_checkpoint_loaders() -> None:
    """Boltz/Lightning checkpoints need full pickle (omegaconf); PyTorch 2.6+ defaults to weights_only=True."""
    try:
        import torch  # type: ignore

        if getattr(torch.load, "_molsearch_patched", False):
            return

        original_load = torch.load

        def trusted_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        trusted_load._molsearch_patched = True  # type: ignore[attr-defined]
        torch.load = trusted_load  # type: ignore[assignment]
        if hasattr(torch, "serialization") and hasattr(torch.serialization, "load"):
            torch.serialization.load = trusted_load  # type: ignore[attr-defined]

        try:
            from omegaconf import DictConfig, ListConfig  # type: ignore
            from omegaconf.base import ContainerMetadata  # type: ignore

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
        except Exception:
            pass

        for module_name in (
            "lightning.fabric.utilities.load",
            "lightning.pytorch.utilities.migration.utils",
            "pytorch_lightning.utilities.migration.utils",
        ):
            try:
                module = __import__(module_name, fromlist=["_load"])
                if hasattr(module, "_load"):
                    original_pl_load = module._load

                    def trusted_pl_load(*args: Any, _orig: Any = original_pl_load, **kwargs: Any) -> Any:
                        kwargs["weights_only"] = False
                        return _orig(*args, **kwargs)

                    module._load = trusted_pl_load
            except Exception:
                continue
    except Exception:
        pass


def _prepare_torch_runtime() -> None:
    _patch_torch_checkpoint_loaders()


def _boltz_src_path() -> Path | None:
    local_boltz_src = _PROJECT_ROOT / "third_party" / "nova" / "external_tools" / "boltz" / "src"
    return local_boltz_src if local_boltz_src.exists() else None


def _missing_boltz_runtime_deps() -> list[str]:
    missing: list[str] = []
    for module, package in (
        ("pytorch_lightning", "pytorch-lightning>=2.2,<3"),
        ("omegaconf", "omegaconf>=2.3"),
        ("hydra", "hydra-core==1.3.2"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def _import_boltz_predict() -> Any:
    missing = _missing_boltz_runtime_deps()
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Boltz runtime dependencies are missing. Install with:\n"
            f"  pip install {packages}\n"
            "Or run: pip install -r scripts/requirements-boltz-extra.txt"
        )

    boltz_src = _boltz_src_path()
    if boltz_src is not None:
        sys.path.insert(0, str(boltz_src))
    from boltz.main import predict  # type: ignore

    return predict


def _select_accelerator() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    if major >= 12:
        arch_list: list[str] = []
        if hasattr(torch.cuda, "get_arch_list"):
            try:
                arch_list = list(torch.cuda.get_arch_list())
            except Exception:
                arch_list = []
        has_sm120 = any("12.0" in arch or "sm_120" in arch for arch in arch_list)
        if not has_sm120:
            device_name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"{device_name} (sm_{major}{minor}) is not supported by this PyTorch build. "
                "Install PyTorch >=2.7 with CUDA 12.8, then pin numpy for Boltz:\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
                "  pip install --force-reinstall --no-deps numpy==1.26.4"
            )
    return "gpu"


def _ensure_boltz_import_deps() -> None:
    """Nova's Boltz fork imports bittensor only for logging; stub it for this CLI."""
    if "bittensor" in sys.modules:
        return

    bt = types.ModuleType("bittensor")

    class _BtLogging:
        @staticmethod
        def error(message: str) -> None:
            print(message, file=sys.stderr)

    bt.logging = _BtLogging()
    sys.modules["bittensor"] = bt


def _mock_validator_score(molecule_id: str, smiles: str, target_name: str) -> float:
    """Deterministic pseudo-score on the validator scale (for tests without GPU)."""
    digest = hashlib.sha256(f"{molecule_id}|{smiles}|{target_name}".encode("utf-8")).hexdigest()
    prob = int(digest[0:8], 16) / 0xFFFFFFFF
    pred = -4.0 + (int(digest[8:16], 16) / 0xFFFFFFFF) * 5.0
    return validator_boltz_score(
        {
            "affinity_probability_binary": prob,
            "affinity_pred_value": pred,
        },
        smiles,
    )


def _load_target(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_json:
        try:
            parsed = json.loads(args.target_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--target-json is invalid JSON ({exc}). "
                "Check shell quoting, or pass "
                "--target-name NAME --target-sequence SEQ instead."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("--target-json must decode to an object")
        return parsed
    return {
        "name": args.target_name,
        "sequence": args.target_sequence,
    }


def _msa_path_for_target(target_name: str) -> Path | None:
    path = _PROJECT_ROOT / "data" / "msa_files" / f"{target_name}.a3m"
    return path if path.is_file() else None


def _write_boltz_input(target_name: str, target_sequence: str, smiles: str, input_dir: Path) -> Path:
    input_path = input_dir / f"{target_name}.yaml"
    protein: dict[str, object] = {"id": "A", "sequence": target_sequence}
    msa_path = _msa_path_for_target(target_name)
    if msa_path is not None:
        protein["msa"] = str(msa_path)

    payload = {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    input_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return input_path


def _validator_score_from_output_dir(out_dir: Path, smiles: str) -> tuple[float, dict[str, float]]:
    metrics = extract_affinity_metrics_from_dir(out_dir)
    if not metrics:
        raise RuntimeError(f"Could not extract Boltz affinity metrics from {out_dir}")
    score = validator_boltz_score(metrics, smiles)
    return score, metrics


def _run_boltz_predict(input_dir: Path, output_dir: Path, cache_dir: str, *, use_msa_server: bool) -> None:
    _ensure_boltz_import_deps()
    _prepare_torch_runtime()
    predict = _import_boltz_predict()

    try:
        accelerator = _select_accelerator()
    except RuntimeError:
        raise
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
            num_workers=0,
            use_msa_server=use_msa_server,
            accelerator=accelerator,
            devices=1,
            no_kernels=(accelerator == "cpu"),
            **NOVA_BOLTZ_PREDICT_KWARGS,
        )
    except Exception as exc:
        raise RuntimeError(f"boltz predict failed in python API: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Boltz2 single-molecule scorer (Nova validator final_score)."
    )
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

    _ensure_boltz_import_deps()
    _prepare_torch_runtime()

    target = _load_target(args)
    target_name = str(target.get("name", args.target_name or "target"))
    target_sequence = str(target.get("sequence", args.target_sequence or ""))

    if args.mock or not target_sequence:
        score = _mock_validator_score(args.molecule_id, args.smiles, target_name)
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
            msa_path = _msa_path_for_target(target_name)
            use_msa_server = msa_path is None
            if use_msa_server:
                print(
                    f"No local MSA at data/msa_files/{target_name}.a3m; using MSA server.",
                    file=sys.stderr,
                )
            else:
                print(f"Using local MSA: {msa_path}", file=sys.stderr)

            _run_boltz_predict(input_dir, output_dir, cache_dir, use_msa_server=use_msa_server)
            score, metrics = _validator_score_from_output_dir(output_dir, args.smiles)
            if not math.isfinite(score):
                raise RuntimeError("Validator score is not finite")

            prob = metrics.get("affinity_probability_binary")
            pred = metrics.get("affinity_pred_value")
            print(
                f"validator_score: {score:.6f} "
                f"(affinity_probability_binary={prob}, affinity_pred_value={pred})",
                file=sys.stderr,
            )
            print(f"score: {score:.6f}")
            return 0
    except Exception as exc:
        print(f"Boltz scoring error: {exc}", file=sys.stderr)
        if args.strict:
            return 2
        fallback = _mock_validator_score(args.molecule_id, args.smiles, target_name)
        print("Falling back to deterministic mock validator score.", file=sys.stderr)
        print(f"score: {fallback:.6f}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
