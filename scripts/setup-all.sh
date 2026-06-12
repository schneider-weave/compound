#!/usr/bin/env bash
# End-to-end setup: venv, deps, data, Boltz, MSA, then optional run.
# Usage:
#   bash scripts/setup-all.sh              # setup only
#   RUN=1 bash scripts/setup-all.sh        # setup + dry-run + score one iteration loop
#   RUN=1 DRY_RUN_ONLY=1 bash scripts/setup-all.sh
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

step "Smoke-test score (validator final_score)"
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

if [[ "${RUN:-0}" == "1" ]]; then
  step "Active search dry-run"
  python -m molsearch.cli dry-run --config config.yaml
  if [[ "${DRY_RUN_ONLY:-0}" != "1" ]]; then
    step "Active search run (generate + score)"
    python -m molsearch.cli run --config config.yaml
    step "Top results"
    python -m molsearch.cli best --config config.yaml --top 20
  fi
fi

echo
echo "Setup complete. Project root: $ROOT"
echo "  source .venv/bin/activate"
echo "  python -m molsearch.cli run --config config.yaml"
