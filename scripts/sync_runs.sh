#!/usr/bin/env bash
# Mirror run artifacts to Google Drive (analysis_plan.md Section 9).
#
# Setup once, on a laptop (never commit the token):
#   rclone config           # create a remote named "gdrive" of type drive
# or:
#   rclone authorize "drive"
#
# Usage:
#   bash scripts/sync_runs.sh            # copy runs/ to gdrive:PromptRepair/runs
#   RCLONE_REMOTE=drive2 bash scripts/sync_runs.sh
#   DRY_RUN=1 bash scripts/sync_runs.sh  # show what would transfer
#
# Uses `copy`, not `sync`: the response cache is append-only, so remote files
# are never deleted. --checksum compares content, not timestamps.
set -euo pipefail

cd "$(dirname "$0")/.."

REMOTE="${RCLONE_REMOTE:-gdrive}"
DEST="${RCLONE_DEST:-PromptRepair/runs}"
SOURCE="${RUNS_DIR:-runs}"
FLAGS=(--checksum --stats-one-line --stats 0)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  FLAGS+=(--dry-run)
fi

mkdir -p "$SOURCE"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Install it and configure a remote named '${REMOTE}'." >&2
  exit 1
fi

if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}:"; then
  echo "rclone remote '${REMOTE}:' is not configured." >&2
  echo "Run 'rclone config' on a machine with a browser, then copy the config." >&2
  exit 1
fi

rclone copy "$SOURCE" "${REMOTE}:${DEST}" "${FLAGS[@]}"

echo "synced ${SOURCE}/ -> ${REMOTE}:${DEST}"
