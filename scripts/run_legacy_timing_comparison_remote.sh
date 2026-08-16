#!/usr/bin/env bash
# Launch legacy_timing_comparison.py detached, for a remote server session
# (e.g. over ssh) that may disconnect long before the legacy pipeline finishes.
#
# Usage:
#   scripts/run_legacy_timing_comparison_remote.sh [--work-dir DIR] [python-script args...]
#
# All arguments are forwarded to legacy_timing_comparison.py. Logs its own
# combined stdout/stderr to <work-dir>/nohup.log (separate from the per-step
# logs legacy_timing_comparison.py already writes under <work-dir>/logs/).

set -euo pipefail

work_dir="benchmark/legacy-timing-comparison"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--work-dir" && $((i + 1)) -lt ${#args[@]} ]]; then
    work_dir="${args[$((i + 1))]}"
  fi
done

mkdir -p "$work_dir"
log_file="$work_dir/nohup.log"
pid_file="$work_dir/nohup.pid"

echo "Logging combined output to $log_file"
nohup uv run python scripts/legacy_timing_comparison.py "$@" >"$log_file" 2>&1 &
pid=$!
echo "$pid" >"$pid_file"
disown "$pid"

echo "Started legacy_timing_comparison.py as PID $pid (recorded in $pid_file)."
echo "You can now safely disconnect. To check on it later:"
echo "  tail -f $log_file"
echo "  kill -0 $pid && echo still running || echo finished"
echo "If it was killed partway through, resume without re-paying for finished steps:"
echo "  scripts/run_legacy_timing_comparison_remote.sh --resume-from $work_dir/benchmark-results.jsonl $*"
