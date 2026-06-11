from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass


SCORE_PATTERNS = [
    re.compile(r"score\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"final_score\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(
        r"affinity_(?:pred_value|probability_binary)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    ),
]


@dataclass(slots=True)
class BoltzScorer:
    mode: str
    command_template: str
    timeout_seconds: int
    mock_score: bool = False
    target: dict[str, object] | None = None

    def score(self, molecule_id: str, smiles: str) -> float:
        if self.mock_score or self.mode.lower() == "mock":
            return self._mock_score(molecule_id, smiles)

        if self.mode.lower() != "command":
            raise ValueError(f"Unsupported scoring mode: {self.mode}")

        target = self.target or {}
        if not target:
            return math.nan

        format_values: dict[str, object] = {
            "smiles": smiles.replace("'", "\\'"),
            "molecule_id": molecule_id.replace("'", "\\'"),
            "target_json": json.dumps(target),
        }
        format_values.update(self._flatten_target_values(target))

        command = self.command_template.format(
            **format_values,
        )

        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except Exception:
            return math.nan

        output = "\n".join([result.stdout or "", result.stderr or ""]).strip()
        return self._extract_score(output)

    @staticmethod
    def _flatten_target_values(
        target: Mapping[str, object],
        prefix: str = "target",
    ) -> dict[str, object]:
        flat: dict[str, object] = {}
        for key, value in target.items():
            safe_key = re.sub(r"[^0-9a-zA-Z_]+", "_", str(key))
            nested_key = f"{prefix}_{safe_key}"
            if isinstance(value, Mapping):
                flat.update(BoltzScorer._flatten_target_values(value, prefix=nested_key))
            else:
                flat[nested_key] = value
        return flat

    @staticmethod
    def _mock_score(molecule_id: str, smiles: str) -> float:
        digest = hashlib.sha256(f"{molecule_id}|{smiles}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        return round(float(value), 6)

    @staticmethod
    def _extract_numeric_from_json(payload: object) -> float | None:
        priority_keys = (
            "score",
            "final_score",
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
        )
        if isinstance(payload, Mapping):
            for key in priority_keys:
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for value in payload.values():
                extracted = BoltzScorer._extract_numeric_from_json(value)
                if extracted is not None:
                    return extracted
        if isinstance(payload, list):
            for value in payload:
                extracted = BoltzScorer._extract_numeric_from_json(value)
                if extracted is not None:
                    return extracted
        return None

    @staticmethod
    def _extract_score(output: str) -> float:
        text = output.strip()
        if not text:
            return math.nan

        # Try JSON first.
        try:
            parsed = json.loads(text)
            extracted = BoltzScorer._extract_numeric_from_json(parsed)
            if extracted is not None:
                return extracted
        except Exception:
            pass

        for pattern in SCORE_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return math.nan

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                return float(line)
            except ValueError:
                continue

        # Last resort: find the last standalone numeric token.
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return math.nan

        return math.nan
