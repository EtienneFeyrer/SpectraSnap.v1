#!/bin/bash
set -euo pipefail

OUT_DIR="precursor_indexes"

echo "[INFO] Creating output directory: $OUT_DIR"
mkdir -p "$OUT_DIR"

export OUT_DIR

extract_tarball() {
    tarball="$1"
    echo "[INFO] Extracting $tarball..."
    tar -xf "$tarball" -C "$OUT_DIR"
}
export -f extract_tarball

echo "[INFO] Extracting tarballs in parallel (4 CPUs)..."

parallel --bar -j 4 extract_tarball ::: batch_*.tar

echo "[INFO] Complete! Extracted $(find "$OUT_DIR" -type f | wc -l) files"