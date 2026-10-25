#!/usr/bin/env bash
# Runs all v0.2 (revision) experiments on a machine with 2 cores, two jobs at a
# time, then aggregates. Every step is cached, so the script can be re-run
# after an interruption. About 6 hours on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")/.."
SCALE=${SCALE:-medium}
OUT=outputs
run() {  # config run_dir
  python scripts/run_experiment.py "configs/experiments/$1.yaml" --scale "$SCALE" --run-dir "$2" --shard 0/2 --threads 1 &
  python scripts/run_experiment.py "configs/experiments/$1.yaml" --scale "$SCALE" --run-dir "$2" --shard 1/2 --threads 1
  wait
  python scripts/run_experiment.py "configs/experiments/$1.yaml" --scale "$SCALE" --run-dir "$2" --threads 2
}
python scripts/prepare_revision_data.py --scale "$SCALE"
run E_factorial "$OUT/E_factorial_$SCALE/run1"
run G_nonpriv "$OUT/G_nonpriv_$SCALE/run1"
run G_search "$OUT/G_search_$SCALE/run1"
run T_timing "$OUT/T_timing_$SCALE/run1"
run L_ablation "$OUT/L_ablation_$SCALE/run1"
run R_reuse "$OUT/R_reuse_$SCALE/run1"
python scripts/eval_scripted.py --out results/revision/scripted_baselines.csv
python scripts/summarize_revision.py --scale "$SCALE" --out results/revision
