#!/bin/bash
set -euo pipefail

TAG="v1.0.0"
OUT_DIR="precursor_indexes"
OWNER="EtienneFeyrer"
REPO="SpectraSnap"

echo "[INFO] Downloading all tarballs using gh release download..."
gh release download "$TAG" \
    --repo "$OWNER/$REPO" \
    --pattern "batch_*.tar" \
    --clobber

echo "[INFO] Verifying downloads..."
for tarball in batch_*.tar; do
    if [ -f "$tarball" ]; then
        size=$(stat -c%s "$tarball" 2>/dev/null || stat -f%z "$tarball" 2>/dev/null)
        size_mb=$((size / 1024 / 1024))
        if [ $size_mb -gt 100 ]; then
            echo "[SUCCESS] $tarball: ${size_mb}MB"
        else
            echo "[ERROR] $tarball is too small: ${size_mb}MB"
            echo "[DEBUG] This might be an HTML error page instead of the actual file"
            echo "[DEBUG] First 200 bytes:"
            head -c 200 "$tarball"
            exit 1
        fi
    fi
done

# Extract
mkdir -p "$OUT_DIR"
echo "[INFO] Extracting tarballs..."
for tarball in batch_*.tar; do
    if [ -f "$tarball" ] && [ -s "$tarball" ]; then
        echo "[INFO] Extracting $tarball..."
        tar -xf "$tarball" -C "$OUT_DIR"
    fi
done

echo "[INFO] Complete! Extracted $(find "$OUT_DIR" -type f | wc -l) files"