#!/usr/bin/env bash
# Run active search: select candidates, score with Boltz, show top results.
# Requires setup first: bash scripts/setup-all.sh
#
# Usage:
#   bash scripts/run.sh                  # dry-run + full run + top 20
#   DRY_RUN_ONLY=1 bash scripts/run.sh   # selection only, no Boltz scoring
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup-all.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -s data/molecules.sqlite ]]; then
  echo "ERROR: data/molecules.sqlite not found. Run: bash scripts/setup-all.sh"
  exit 1
fi

CONFIG="${CONFIG:-config.yaml}"
step() { echo; echo "=== $* ==="; }

step "Dry-run (select batch, no scoring)"
python -m molsearch.cli dry-run --config "$CONFIG"

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo
  echo "Dry-run only — skipping Boltz scoring."
  exit 0
fi

step "Run (generate + score)"
python -m molsearch.cli run --config "$CONFIG"

step "Top results"
python -m molsearch.cli best --config "$CONFIG" --top 20
