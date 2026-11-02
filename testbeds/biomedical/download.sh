#!/usr/bin/env bash
# Download the Synthea synthetic patient sample (CSV, Nov 2021) into data/raw/synthea.
# Synthea data is fully synthetic. Source: github.com/synthetichealth/synthea-sample-data
# Fetches only the one zip file instead of the whole repository.
set -euo pipefail
DEST="${1:-data/raw/synthea}"
FILE="downloads/synthea_sample_data_csv_nov2021.zip"
TMP="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/synthetichealth/synthea-sample-data.git "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set --no-cone "$FILE"
git -C "$TMP/repo" checkout
mkdir -p "$DEST"
unzip -o -q "$TMP/repo/$FILE" -d "$DEST"
rm -rf "$TMP"
echo "Synthea CSV saved to $DEST/csv"
ls -lh "$DEST/csv"
