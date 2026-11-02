#!/usr/bin/env bash
# Download CUAD v1 (CC BY 4.0, The Atticus Project) into data/raw/cuad.
set -euo pipefail
DEST="${1:-data/raw/cuad}"
TMP="$(mktemp -d)"
git clone --depth 1 https://github.com/TheAtticusProject/cuad.git "$TMP/cuad"
mkdir -p "$DEST"
unzip -o -q "$TMP/cuad/data.zip" -d "$DEST"
cp "$TMP/cuad/category_descriptions.csv" "$DEST/"
rm -rf "$TMP"
echo "CUAD saved to $DEST"
ls -lh "$DEST"
