#!/bin/bash
# Full speckle experiment, resumable: tune -> stop if a chosen parameter is on its grid edge
# -> run (1493 test scenes, 100 with seeds 1-4 at SSR 5/0 dB) -> eval -> report.
cd /home/d14/d14k42007/csa-sar-imaging || exit 1
R=full; D=end_to_end_pipeline/speckle/$R
echo "=== start $(date)"
taskset -c 0,2-23 python -m sarsim.speckle_pipeline tune --run $R || { echo "=== tune FAILED $(date)"; exit 1; }
echo "=== tune done $(date)"
taskset -c 0,2-23 python - <<'PY' || { echo "=== STOP: parameter on grid edge, extend the grid $(date)"; exit 2; }
import json, sys
e = json.load(open("end_to_end_pipeline/speckle/full/chosen_params.json"))["on_grid_edge"]
print("on_grid_edge:", e)
sys.exit(1 if e else 0)
PY
taskset -c 0,2-23 python -m sarsim.speckle_pipeline run --run $R --var-n 100 || { echo "=== run FAILED $(date)"; exit 1; }
echo "=== run done $(date)"
taskset -c 0,2-23 python -m sarsim.speckle_pipeline eval --run $R || { echo "=== eval FAILED $(date)"; exit 1; }
taskset -c 0,2-23 python -m sarsim.speckle_pipeline report --run $R
echo "=== all done $(date)"
