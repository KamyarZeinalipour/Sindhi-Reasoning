#!/usr/bin/env bash
# Automatic metrics for every file in results/raw.
set -euo pipefail
cd "$(dirname "$0")/.."

python evaluation/metrics.py results/raw -o results/metrics
