# Molecule Active Search

Compute-efficient molecule search for Nova subnet reactions **1, 2, and 4**. You fix one combinatorial parameter; the pipeline searches the remaining dimension(s) with duplicate-aware active learning and Boltz-2 scoring aligned to the validator `final_score`.

## How It Works

1. **Search space** — Filter `molecules.sqlite` to your reaction and fixed param (e.g. `rxn:1` with `p1=60111` → 1D search over `p2`).
2. **Product SMILES** — Build `molecule_id` from the DB; resolve the **product SMILES** via combinatorial chemistry (same logic as the Nova validator), not reactant concatenation.
3. **Active learning** — Train on history, select exploit + UCB + explore batches, skip already-scored IDs/SMILES.
4. **Scoring** — Run Boltz-2 through `score_boltz2.py`, which outputs validator `final_score`:

```text
(affinity_probability_binary - affinity_pred_value) / heavy_atom_count
```

## Project Layout

- `config.yaml` — reaction, fixed params, scoring target
- `data/molecules.sqlite` — Hugging Face combinatorial DB
- `data/nova_results_Molecules_RXN*.csv` — origin/history scores
- `data/my_results_rxn*.csv` — your new scored rows
- `data/msa_files/` — validator MSA (optional, improves score match)
- `src/molsearch/reactions.py` — SMILES from `molecule_id`
- `score_boltz2.py` — Boltz CLI scorer (validator formula)

## Quick Start (two steps)

**Step 1 — clone + install environment** (venv, DB, Boltz, MSA):

```bash
git clone https://github.com/schneider-weave/compound.git molecule-active-search && \
cd molecule-active-search && \
bash scripts/setup-all.sh
```

**Step 2 — run pipeline** (select candidates + score):

```bash
source .venv/bin/activate
bash scripts/run.sh
```

Selection only, no Boltz:

```bash
DRY_RUN_ONLY=1 bash scripts/run.sh
```

## Installation

```bash
cd molecule-active-search
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional CatBoost model:

```bash
pip install -e .[catboost]
```

## Download Assets

Combinatorial database:

```bash
curl -L "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite" \
  -o data/molecules.sqlite
```

Nova Boltz tool (real scoring):

```bash
mkdir -p third_party
git clone --filter=blob:none --sparse https://github.com/metanova-labs/nova.git third_party/nova
git -C third_party/nova sparse-checkout set external_tools/boltz
bash scripts/setup-boltz-gpu.sh
```

Validator MSA (recommended for ~subnet score match):

```bash
bash scripts/fetch-nova-msa.sh Q4QQW4
```

## Configure

Example for **rxn 1** with fixed param `p1=60111` (1D search over `p2`):

```yaml
search:
  rxn: 1
  fixed_params:
    1: 60111
  batch_size: 60
```

Same pattern for rxn 2 and 4. For rxn 3/5 with two fixed params, search reduces to 1D.

Set `files.molecules_sqlite` or env `MOLECULES_SQLITE` if the DB is not at `data/molecules.sqlite`.

## Run

Dry run (selection only, no Boltz):

```bash
python -m molsearch.cli dry-run --config config.yaml
```

Score one batch:

```bash
python -m molsearch.cli run --config config.yaml
```

Inspect candidates or top scores:

```bash
python -m molsearch.cli candidates --config config.yaml
python -m molsearch.cli best --config config.yaml --top 20
```

## Scoring Config

```yaml
scoring:
  mode: "command"
  command_template: "env BOLTZ_CACHE=data/boltz-cache python3 score_boltz2.py {strict_flag} --smiles '{smiles}' --molecule-id '{molecule_id}' --target-json '{target_json}'"
  target:
    name: "Q4QQW4"
    sequence: "..."
  timeout_seconds: 3600
```

Template variables: `{molecule_id}`, `{smiles}`, `{target_json}`, `{target_name}`, `{target_sequence}`, etc.

## Result Files

My results schema:

```csv
molecule_id,smiles,final_score
```

Writes are atomic; duplicate molecule IDs keep the latest row; results sort by `final_score` descending.
