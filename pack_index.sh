#!/bin/bash
set -euo pipefail

SRC_DIR="precursor_indexes"
OUT_DIR="precursor_index_tarballs"
TMP_DIR="$OUT_DIR/batches"
MAX_SIZE_BYTES=$((2 * 1024 * 1024 * 1024 - 200 * 1024 * 1024))  # 2GB - 200MB safety margin

mkdir -p "$OUT_DIR" "$TMP_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# ------------------------------------------------------------
# NEW: Skip Step 1 + Step 2 if batches already exist
# ------------------------------------------------------------
if ls "$TMP_DIR"/batch_*.txt >/dev/null 2>&1; then
    log "Existing batches detected — skipping file listing and batching."
else
    log "No batches found — running file listing and batching."

    log "Step 1: Building file list with sizes..."
    find "$SRC_DIR" -type f -printf "%s %p\n" > "$TMP_DIR/files_with_sizes.txt"

    TOTAL=$(wc -l < "$TMP_DIR/files_with_sizes.txt")
    log "Found $TOTAL files."

    log "Step 2: Grouping files into batches under 2GB..."

    BATCH_NUM=1
    CURRENT_BATCH="$TMP_DIR/batch_$(printf "%03d" $BATCH_NUM).txt"
    CURRENT_SIZE=0
    COUNT=0

    > "$CURRENT_BATCH"

    while IFS= read -r line; do
        COUNT=$((COUNT+1))
        (( COUNT % 5000 == 0 )) && log "Batching progress: $COUNT / $TOTAL files"

        size="${line%% *}"
        file="${line#* }"

        if (( CURRENT_SIZE + size > MAX_SIZE_BYTES )) && (( CURRENT_SIZE > 0 )); then
            BATCH_NUM=$((BATCH_NUM + 1))
            CURRENT_BATCH="$TMP_DIR/batch_$(printf "%03d" $BATCH_NUM).txt"
            > "$CURRENT_BATCH"
            CURRENT_SIZE=0
            log "Starting batch $BATCH_NUM..."
        fi

        echo "$file" >> "$CURRENT_BATCH"
        CURRENT_SIZE=$((CURRENT_SIZE + size))

    done < "$TMP_DIR/files_with_sizes.txt"

    log "Created $BATCH_NUM batches."
fi

# ------------------------------------------------------------
# Step 3: Tar creation (always runs)
# ------------------------------------------------------------

log "Step 3: Creating tarballs in parallel (50 cores)..."

export OUT_DIR

create_tarball() {
    batch_file="$1"
    batch_name=$(basename "$batch_file" .txt)
    tarball="$OUT_DIR/${batch_name}.tar"

    echo "[$(date '+%H:%M:%S')] Packing $tarball ..."
    tar --no-recursion -cf "$tarball" -T "$batch_file"

    actual_size=$(stat -c "%s" "$tarball")
    size_gb=$(echo "scale=2; $actual_size / 1024 / 1024 / 1024" | bc)

    if (( actual_size > 2*1024*1024*1024 )); then
        echo "[$(date '+%H:%M:%S')] WARNING: $tarball is ${size_gb}GB (exceeds 2GB!)"
    else
        echo "[$(date '+%H:%M:%S')] OK: $tarball is ${size_gb}GB"
    fi
}

export -f create_tarball

log "Using GNU parallel (50 cores)."

parallel --bar -j 50 create_tarball ::: "$TMP_DIR"/batch_*.txt

log "All tarballs created in $OUT_DIR"
ls -lh "$OUT_DIR"/*.tar
