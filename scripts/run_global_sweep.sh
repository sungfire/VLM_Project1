#!/bin/bash
# usage: run_global_sweep.sh <device> <tokens...>
cd "$(dirname "$0")/.."
dev=$1; shift
for t in "$@"; do
  .venv/bin/python -m src.run_eval --config configs/baselines/global.yaml --device $dev --set name=global_$t tokens=$t
done
