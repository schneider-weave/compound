#!/usr/bin/env bash
# RTX 5060 Ti (sm_120) + Boltz 2.2.0 environment setup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f ../.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../.venv/bin/activate
  cd ..
fi

echo "=== Step 1: remove old PyTorch (cu124/cu126 lacks sm_120) ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "=== Step 2: install PyTorch with CUDA 12.8 (Blackwell / sm_120) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 3: install Boltz package + runtime dependencies ==="
if [[ ! -d third_party/nova/external_tools/boltz/src/boltz ]]; then
  echo "ERROR: Boltz source not found. Run:"
  echo "  mkdir -p third_party"
  echo "  git clone --filter=blob:none --sparse https://github.com/metanova-labs/nova.git third_party/nova"
  echo "  git -C third_party/nova sparse-checkout set external_tools/boltz"
  exit 1
fi

pip install -r third_party/nova/external_tools/boltz/requirements.txt
pip install -r scripts/requirements-boltz-extra.txt
pip install -e third_party/nova/external_tools/boltz

echo "=== Step 4: pin deps required by boltz 2.2.0 ==="
pip install --force-reinstall --no-deps numpy==1.26.4 scipy==1.13.1 numba==0.61.0

echo "=== Step 5: verify GPU + checkpoint patch prerequisites ==="
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    arch = list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    print("supported arch:", arch)
    if not any("12.0" in a or "sm_120" in a for a in arch):
        raise SystemExit(
            "ERROR: this PyTorch build still lacks sm_120. "
            "Re-run with: pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    x = torch.ones(1, device="cuda")
    print("cuda smoke test:", float(x.sum()))

from omegaconf import DictConfig
import torch.serialization
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([DictConfig])
print("checkpoint globals: ok")
PY

echo
echo "Setup complete. Test scoring with:"
echo "  cd $ROOT"
echo '  env BOLTZ_CACHE=data/boltz-cache python3 score_boltz2.py --strict \'
echo '    --smiles "NC(c1cn(I)nn1)c1cccs1" \'
echo '    --molecule-id "rxn:1:60111:39851" \'
echo '    --target-name Q4QQW4 \'
echo '    --target-sequence MAQTQGTKRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEEDPDKRISICSSDKRIACEEEFSDSDEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKMA'
