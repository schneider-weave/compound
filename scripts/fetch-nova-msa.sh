#!/usr/bin/env bash
# Download Nova validator MSA files for Boltz scoring.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-Q4QQW4}"
OUT_DIR="$ROOT/data/msa_files"
OUT_FILE="$OUT_DIR/${TARGET}.a3m"
URL="https://raw.githubusercontent.com/metanova-labs/nova/main/data/msa_files/${TARGET}.a3m"

mkdir -p "$OUT_DIR"
echo "Downloading $URL"
curl -fsSL "$URL" -o "$OUT_FILE"
echo "Saved to $OUT_FILE ($(wc -l < "$OUT_FILE") lines)"
