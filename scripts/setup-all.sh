#!/usr/bin/env bash
# Environment setup only: venv, deps, data, Boltz, MSA.
# Does NOT run the active-search pipeline — use scripts/run.sh for that.
#
# Usage:
#   bash scripts/setup-all.sh
#   SKIP_SMOKE_TEST=1 bash scripts/setup-all.sh   # skip single-molecule Boltz check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { echo; echo "=== $* ==="; }

step "Python venv"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .

step "Combinatorial DB (molecules.sqlite)"
mkdir -p data
if [[ ! -s data/molecules.sqlite ]]; then
  curl -fsSL \
    "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite" \
    -o data/molecules.sqlite
else
  echo "Already present: data/molecules.sqlite"
fi

step "Nova Boltz source"
if [[ ! -d third_party/nova/external_tools/boltz/src/boltz ]]; then
  mkdir -p third_party
  git clone --filter=blob:none --sparse https://github.com/metanova-labs/nova.git third_party/nova
  git -C third_party/nova sparse-checkout set external_tools/boltz
else
  echo "Already present: third_party/nova/external_tools/boltz"
fi

step "GPU + Boltz install"
bash scripts/setup-boltz-gpu.sh

step "Validator MSA"
bash scripts/fetch-nova-msa.sh Q4QQW4

if [[ "${SKIP_SMOKE_TEST:-0}" != "1" ]]; then
  step "Smoke-test score (verify Boltz env)"
  TARGET_SEQ="$(python3 - <<'PY'
import yaml
print(yaml.safe_load(open("config.yaml"))["scoring"]["target"]["sequence"])
PY
)"
  env BOLTZ_CACHE=data/boltz-cache python3 score_boltz2.py --strict \
    --smiles "CC1(C(=O)c2cn(I)nn2)COCC1N" \
    --molecule-id "rxn:1:60111:2212" \
    --target-name Q4QQW4 \
    --target-sequence "$TARGET_SEQ"
fi

echo
echo "Environment setup complete: $ROOT"
echo
echo "Next — run the pipeline:"
echo "  source .venv/bin/activate"
echo "  bash scripts/run.sh"
