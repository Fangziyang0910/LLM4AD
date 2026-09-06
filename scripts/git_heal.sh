#!/usr/bin/env bash
# Automatically detect and heal corrupted Git index file (.git/index)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INDEX_FILE=".git/index"
LOCK_FILE=".git/index.lock"

# 1. Check if index.lock is stale (not being written by any active git process)
if [ -f "$LOCK_FILE" ]; then
    if ! pgrep -f "git " >/dev/null 2>&1; then
        echo "Found stale $LOCK_FILE without running git process. Removing..."
        rm -f "$LOCK_FILE"
    fi
fi

# 2. Check if .git/index is 0 bytes or invalid
IS_CORRUPT=0
if [ ! -f "$INDEX_FILE" ]; then
    IS_CORRUPT=1
elif [ ! -s "$INDEX_FILE" ]; then
    echo "Detected 0-byte .git/index file."
    IS_CORRUPT=1
elif ! git status >/dev/null 2>&1; then
    ERR=$(git status 2>&1 || true)
    if echo "$ERR" | grep -qE "(smaller than expected|corrupt|bad index)"; then
        echo "Detected corrupted .git/index: $ERR"
        IS_CORRUPT=1
    fi
fi

if [ "$IS_CORRUPT" -eq 1 ]; then
    echo "Healing .git/index..."
    rm -f "$INDEX_FILE"
    git reset
    echo "✅ .git/index has been successfully restored from HEAD!"
    if [ -s "${INDEX_FILE}.bak" ] && [ "$(stat -c %s "${INDEX_FILE}.bak")" -gt 100 ]; then
        cp "${INDEX_FILE}.bak" "$INDEX_FILE"
        if git status >/dev/null 2>&1; then
            echo "✅ .git/index restored instantly from shadow backup!"
        else
            rm -f "$INDEX_FILE"
            git reset
            echo "✅ .git/index has been successfully restored from HEAD via git reset!"
        fi
    else
        git reset
        echo "✅ .git/index has been successfully restored from HEAD via git reset!"
    fi
    git status -s
else
    # Keep shadow backup fresh
    cp -p "$INDEX_FILE" "${INDEX_FILE}.bak" 2>/dev/null || true
    echo "✅ .git/index is healthy ($(stat -c %s "$INDEX_FILE") bytes)."
fi

