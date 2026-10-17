#!/bin/bash
# Full thermal-noise experiment, resumable: tune (per SNR level, grids auto-extended on edge picks)
# -> stop if a choice is still on an edge -> run (1493 test scenes; 100 with noise seeds 1-4 at 5/0 dB)
# -> eval -> report. Pinned away from the faulty CPU 1.
cd /home/d14/d14k42007/csa-sar-imaging || exit 1
R=full
P="taskset -c 0,2-6,8-23 python -m sarsim.noise_pipeline"
echo "=== start $(date)"
$P tune --run $R || { echo "=== tune FAILED $(date)"; exit 1; }
echo "=== tune done $(date)"
taskset -c 0,2-6,8-23 python - <<'PY' || { echo "=== STOP: choice still on a grid edge $(date)"; exit 2; }
import json, sys
c = json.load(open("end_to_end_pipeline/noise/full/chosen_params.json"))["per_level"]
e = {lv: {m: d["on_edge"] for m, d in ms.items() if d["on_edge"]} for lv, ms in c.items()}
e = {k: v for k, v in e.items() if v}
print("on_edge:", e)
sys.exit(1 if e else 0)
PY
$P run --run $R --var-n 100 || { echo "=== run FAILED $(date)"; exit 1; }
echo "=== run done $(date)"
$P eval --run $R || { echo "=== eval FAILED $(date)"; exit 1; }
$P report --run $R
echo "=== all done $(date)"
