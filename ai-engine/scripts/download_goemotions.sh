#!/usr/bin/env bash
# Download the GoEmotions train/dev/test splits (Apache-2.0) into data/goemotions/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/goemotions
base="https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
for split in train dev test; do
  curl -fsSL "$base/$split.tsv" -o "data/goemotions/$split.tsv"
done
wc -l data/goemotions/*.tsv
