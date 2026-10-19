#!/usr/bin/env python3
"""Thermal-noise sweep: unmasked scene (no speckle, a_k = A_k) -> exact echo
(sarsim.echo.gen_echo_signals) -> CSA -> + receiver noise at several SNRs ->
CSA / L0 / L1 / L1/2&TV (sarsim/refine.py) -> 30 dB JPG -> YOLO P / R / mAP@0.5.

Noise model (midterm report: Y = S + W): W is i.i.d. CN(0, sigma_W^2) receiver
noise added to the echo. The CSA is unitary, so CSA(W) is again i.i.d.
CN(0, sigma_W^2) and the noise is added to the focused image directly (exact,
checked by `validate`). The level is

    SNR = 10 log10( mean_scene |H_S|^2 / sigma^2 )   (dB),

mean over the 800 x 800 scene of the noise-free CSA image. One unit-variance
field w ~ CN(0, 1) per scene (SeedSequence([seed, crc32(stem), 1])) is scaled
by sigma for every level, so all levels and methods of a scene share it.

Parameters (two protocols, both chosen on the 100 tune scenes, never on test):
    C  per level: each SNR level gets its own grid search (the receiver noise
       power is a calibrated, known system parameter)
    A  robustness: the parameters chosen at SNR = inf applied at every level
Grid search maximizes tune-set mAP@0.5 (ties within 0.002: higher F1, then the
simpler setting). A choice on the edge of its grid extends the grid on that side
and is searched again (up to --max-rounds).

Steps (end_to_end_pipeline/noise/<run>/):
    split | validate | tune | run | eval | report      (see each cmd_* below)
"""
import argparse
import csv
import itertools
import json
import os
import time
import zlib

import numpy as np

from .pipeline import exclude_cpus
from .pipeline import git_commit
from .speckle_pipeline import (ALPHAS_THRESHOLD, IMAGES, MARGIN, METHODS, PIPE, Context, IterLog, Timer, config_name,
                               iteration_sample, summarize_iterations, to_gray)

SNR_LEVELS = [float("inf"), 10.0, 5.0, 0.0, -5.0]
GRID0 = {
    "l0": [{"alpha": a} for a in ALPHAS_THRESHOLD],
    "l1": [{"alpha": a} for a in ALPHAS_THRESHOLD],
    "l12tv": [{"alpha": a, "xi": x, "tv": t}
              for a, x, t in itertools.product((0.08, 0.13, 0.20), (1.0, 2.0, 3.0), (0.01, 0.03, 0.1))],
}
# grid extension when a choice sits on an edge: multiply (upper edge) / divide (lower edge)
EXTEND = {"alpha": 1.25, "xi": 1.5, "tv": 3.0}


def snr_name(v):
    return "snr" + ("inf" if v == float("inf") else f"{v:g}")


def run_dir(args):
    return os.path.join(PIPE, "noise", args.run)


def noise_field(stem, shape, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(stem.encode()), 1]))
    re, im = rng.standard_normal((2,) + tuple(shape))
    return (re + 1j * im) / np.sqrt(2.0)


def clean_crop(ctx, amp):
    """Noise-free CSA image (scene + MARGIN crop) of the scene amplitude, exact echo."""
    from .echo import gen_echo_signals
    from .params import build_point_target_list
    from .speckle import coef_sets
    az, rg, _ = build_point_target_list(amp, ctx.ax["n_row"], ctx.ax["n_col"],
                                        ctx.ax["pulse_rep_freq_hz"], ctx.ax["sampling_freq_hz"])
    echo = gen_echo_signals(ctx.ax, az, rg, coef_sets(amp, []))
    crop = ctx.crop(ctx.csa.focus(echo[0])).copy()
    del echo
    return crop


def scene_power(cp, crop):
    s = crop[MARGIN:-MARGIN, MARGIN:-MARGIN]
    return float(cp.mean(s.real ** 2 + s.imag ** 2))


def noisy(cp, crop, w, snr, power):
    if snr == float("inf"):
        return crop
    return crop + float(np.sqrt(power / 10.0 ** (snr / 10.0))) * w


def load_split(args):
    with open(os.path.join(run_dir(args), "split.json")) as f:
        return json.load(f)


def write_manifest(args, extra):
    import datetime
    path = os.path.join(run_dir(args), "manifest.json")
    m = {}
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
    m.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    m.update({"git_commit": git_commit(), "numpy": np.__version__, "base_seed": args.seed, "margin": MARGIN,
              "echo": "sarsim.echo.gen_echo_signals (FP64 geometry, FP32 chirp loop)", **extra})
    with open(path, "w") as f:
        json.dump(m, f, indent=2)


def cmd_split(args):
    """Same tune / test split as sarsim.speckle_pipeline (seed 0, 100 tune scenes)."""
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    rng = np.random.default_rng(args.split_seed)
    tune = sorted(rng.choice(stems, args.n_tune, replace=False).tolist())
    test = sorted(set(stems) - set(tune))
    os.makedirs(run_dir(args), exist_ok=True)
    with open(os.path.join(run_dir(args), "split.json"), "w") as f:
        json.dump({"seed": args.split_seed, "tune": tune, "test": test}, f, indent=1)
    print(f"{len(tune)} tune / {len(test)} test scenes -> {run_dir(args)}/split.json")


def cmd_validate(args):
    """CSA(W) for echo-domain white noise W: variance preserved (unitary) and white."""
    import cupy as cp
    ctx = Context(cp)
    rng = cp.random.RandomState(0)
    shape = (ctx.ax["n_row"], ctx.ax["n_col"])
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    h = ctx.csa.focus(w)
    c = h[MARGIN:-MARGIN, MARGIN:-MARGIN]
    var_ratio = float(cp.mean(cp.abs(h) ** 2) / cp.mean(cp.abs(w) ** 2))
    lag = lambda a, b: abs(complex(cp.mean(a * b.conj()))) / float(cp.mean(cp.abs(c) ** 2))
    stem = load_split(args)["test"][0]
    amp = ctx.load_amp(stem)
    crop = clean_crop(ctx, amp)
    p = scene_power(cp, crop)
    wn = cp.asarray(noise_field(stem, crop.shape, args.seed))
    lines = ["CSA of echo-domain white noise CN(0,1) on the full grid:",
             f"  output / input variance = {var_ratio:.6f} (unitary: 1)",
             f"  |correlation| at lag 1: azimuth {lag(c[1:, :], c[:-1, :]):.4f}, range {lag(c[:, 1:], c[:, :-1]):.4f} (white: 0)",
             "", f"scene {stem}: mean scene |H_S|^2 = {p:.4g}", "",
             "| SNR (dB) | added noise power / scene power (dB) |", "|---|---|"]
    for v in SNR_LEVELS[1:]:
        h_n = noisy(cp, crop, wn, v, p)
        d = (h_n - crop)[MARGIN:-MARGIN, MARGIN:-MARGIN]
        lines.append(f"| {v:g} | {10 * np.log10(float(cp.mean(cp.abs(d) ** 2)) / p):.3f} |")
    os.makedirs(run_dir(args), exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "validate.md"), "w") as f:
        f.write(text)
    print(text)


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {snr_name(v): {m: list(g) for m, g in GRID0.items()} for v in SNR_LEVELS}


def choose(rows, method, grid):
    cand = [r for r in rows if r["method"] == method]
    best = max(r["map50"] for r in cand)
    near = [r for r in cand if r["map50"] >= best - 0.002]
    best_f1 = max(r["f1"] for r in near)
    near = [r for r in near if r["f1"] >= best_f1 - 1e-9]
    pick = min(near, key=lambda r: (r["params"].get("alpha", 0), r["params"].get("tv", 0), r["params"].get("xi", 0)))
    edges = []
    for k, v in pick["params"].items():
        values = sorted({p[k] for p in grid})
        if v == values[-1]:
            edges.append((k, "up"))
        elif v == values[0]:
            edges.append((k, "down"))
    return pick, edges


def extend(grid, edges):
    """New configs beyond each edge, crossed with the other parameters' current values."""
    new = []
    keys = list(grid[0].keys())
    for k, side in edges:
        values = sorted({p[k] for p in grid})
        v = values[-1] * EXTEND[k] if side == "up" else values[0] / EXTEND[k]
        others = [sorted({p[o] for p in grid}) if o != k else [round(v, 6)] for o in keys]
        new += [dict(zip(keys, combo)) for combo in itertools.product(*others)]
    seen = {json.dumps(p, sort_keys=True) for p in grid}
    return [p for p in new if json.dumps(p, sort_keys=True) not in seen]


def cmd_tune(args):
    import cupy as cp
    import cv2
    from .detect_eval import evaluate, load_model
    from .refine import apply
    ctx = Context(cp)
    stems = load_split(args)["tune"][: args.n]
    out = os.path.join(run_dir(args), "tune")
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "tune")
    iters = IterLog(os.path.join(run_dir(args), "iterations.csv"), "tune")
    state_path = os.path.join(run_dir(args), "tune_grid.json")
    grids = load_state(state_path)
    results_path = os.path.join(run_dir(args), "tune_results.csv")
    evaluated = {}
    if os.path.exists(results_path):
        for r in csv.DictReader(open(results_path)):
            evaluated[(r["snr"], r["config"])] = r
    model = load_model()

    def configs(level):
        return [("csa", {})] + [(m, p) for m in ("l0", "l1", "l12tv") for p in grids[level][m]]

    for rnd in range(args.max_rounds):
        # images for every (level, config) not yet on disk
        for i, stem in enumerate(stems):
            todo = {snr_name(v): [(m, p) for m, p in configs(snr_name(v))
                                  if not os.path.exists(os.path.join(out, snr_name(v), config_name(m, p), stem + ".jpg"))]
                    for v in SNR_LEVELS}
            if not any(todo.values()):
                continue
            t0 = time.perf_counter()
            amp = ctx.load_amp(stem)
            crop = clean_crop(ctx, amp)
            p = scene_power(cp, crop)
            w = cp.asarray(noise_field(stem, crop.shape, args.seed))
            cp.cuda.Stream.null.synchronize()
            timer.log(stem, "echo+csa", time.perf_counter() - t0)
            for v in SNR_LEVELS:
                h = noisy(cp, crop, w, v, p)
                for m, prm in todo[snr_name(v)]:
                    t0 = time.perf_counter()
                    img, info = apply(cp, m, h, prm, MARGIN)
                    d = os.path.join(out, snr_name(v), config_name(m, prm))
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, stem + ".jpg"), to_gray(cp, img))
                    timer.log(stem, m, time.perf_counter() - t0)
                    iters.log(stem, args.seed, snr_name(v), m, config_name(m, prm), info, time.perf_counter() - t0)
            del crop, w
            cp.get_default_memory_pool().free_all_blocks()
            print(f"  tune round {rnd + 1} [{i + 1}/{len(stems)}] {stem}", flush=True)
        # YOLO on every (level, config) not yet evaluated
        t0 = time.perf_counter()
        n_eval = 0
        for v in SNR_LEVELS:
            lv = snr_name(v)
            for m, prm in configs(lv):
                name = config_name(m, prm)
                if (lv, name) in evaluated:
                    continue
                r = evaluate(model, os.path.join(out, lv, name), stems,
                             os.path.join(run_dir(args), "_eval", "tune", lv, name))
                evaluated[(lv, name)] = {"snr": lv, "config": name, "method": m, **{k: str(x) for k, x in prm.items()},
                                         **{k: f"{x:.6f}" for k, x in r.items()}}
                n_eval += 1
        timer.log("-", f"yolo_eval_{n_eval * len(stems)}", time.perf_counter() - t0)
        keys = ["snr", "config", "method", "alpha", "xi", "tv", "precision", "recall", "f1", "map50"]
        with open(results_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(sorted(evaluated.values(), key=lambda r: (r["snr"], r["config"])))
        # choose per level; extend grids whose choice is on an edge
        chosen, any_edge = {}, False
        for v in SNR_LEVELS:
            lv = snr_name(v)
            chosen[lv] = {}
            for m in ("l0", "l1", "l12tv"):
                rows = [{"method": m, "params": prm, "map50": float(evaluated[(lv, config_name(m, prm))]["map50"]),
                         "f1": float(evaluated[(lv, config_name(m, prm))]["f1"])} for prm in grids[lv][m]]
                pick, edges = choose(rows, m, grids[lv][m])
                chosen[lv][m] = {"params": pick["params"], "map50": pick["map50"], "on_edge": edges}
                new = extend(grids[lv][m], edges) if edges else []
                if new:
                    grids[lv][m] += new
                    any_edge = True
                    print(f"  {lv} {m}: {pick['params']} on edge {edges} -> +{len(new)} configs", flush=True)
        with open(state_path, "w") as f:
            json.dump(grids, f, indent=1)
        with open(os.path.join(run_dir(args), "chosen_params.json"), "w") as f:
            json.dump({"n_tune": len(stems), "round": rnd + 1, "per_level": chosen}, f, indent=2)
        if not any_edge:
            break
    write_manifest(args, {"experiment": "thermal noise", "snr_levels_db": [str(v) for v in SNR_LEVELS],
                          "n_tune": len(stems), "chosen_params": chosen})
    print(json.dumps(chosen, indent=1))


def load_params(args):
    with open(os.path.join(run_dir(args), "chosen_params.json")) as f:
        per = json.load(f)["per_level"]
    c = {lv: {"csa": {}, **{m: d["params"] for m, d in per[lv].items()}} for lv in per}
    return c, c[snr_name(float("inf"))]


def cmd_run(args):
    import cupy as cp
    import cv2
    from .refine import apply
    ctx = Context(cp)
    params_c, params_a = load_params(args)
    stems = load_split(args)["test"][: args.n]
    var_stems = set(stems[: args.var_n])
    var_seeds = [int(v) for v in args.var_seeds.split(",") if v.strip()]
    var_snr = [float(v) for v in args.var_snr.split(",") if v.strip()]
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "run")
    iters = IterLog(os.path.join(run_dir(args), "iterations.csv"), "run")
    write_manifest(args, {"n_test": len(stems), "var_n": len(var_stems), "var_seeds": var_seeds,
                          "var_snr_db": var_snr, "params_C": params_c, "params_A": params_a})
    base = os.path.join(run_dir(args), "test")

    def jobs(stem):
        """(seed, level, output method dir, refine method, params)"""
        out = []
        for v in SNR_LEVELS:
            lv = snr_name(v)
            for m in METHODS:
                out.append((args.seed, v, m, m, params_c[lv][m]))
                if m != "csa" and v != float("inf"):
                    out.append((args.seed, v, m + "_tunedinf", m, params_a[m]))
        if stem in var_stems:
            out += [(sd, v, m, m, params_c[snr_name(v)][m]) for sd in var_seeds for v in var_snr for m in METHODS]
        return out

    for i, stem in enumerate(stems):
        js = jobs(stem)
        paths = [os.path.join(base, f"seed{sd}", snr_name(v), d, stem + ".jpg") for sd, v, d, _, _ in js]
        if all(os.path.exists(p) for p in paths):
            continue
        t0 = time.perf_counter()
        amp = ctx.load_amp(stem)
        crop = clean_crop(ctx, amp)
        p = scene_power(cp, crop)
        cp.cuda.Stream.null.synchronize()
        timer.log(stem, "echo+csa", time.perf_counter() - t0)
        fields = {}
        for (sd, v, d, m, prm), path in zip(js, paths):
            if os.path.exists(path):
                continue
            if sd not in fields:
                fields[sd] = cp.asarray(noise_field(stem, crop.shape, sd))
            t0 = time.perf_counter()
            img, info = apply(cp, m, noisy(cp, crop, fields[sd], v, p), prm, MARGIN)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, to_gray(cp, img))
            timer.log(stem, m, time.perf_counter() - t0)
            iters.log(stem, sd, snr_name(v), m, d, info, time.perf_counter() - t0)
        del crop, fields
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  run [{i + 1}/{len(stems)}] {stem}", flush=True)


def cmd_eval(args):
    from .detect_eval import evaluate, load_model
    with open(os.path.join(run_dir(args), "manifest.json")) as f:
        man = json.load(f)
    test = load_split(args)["test"]
    base = os.path.join(run_dir(args), "test")
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "eval")
    model = load_model()

    def ev(seed, v, d, stems, tag):
        path = os.path.join(base, f"seed{seed}", snr_name(v), d)
        stems = [s for s in stems if os.path.exists(os.path.join(path, s + ".jpg"))]
        t0 = time.perf_counter()
        r = evaluate(model, path, stems, os.path.join(run_dir(args), "_eval", tag, f"seed{seed}", snr_name(v), d))
        timer.log("-", f"yolo_eval_{len(stems)}", time.perf_counter() - t0)
        return {"set": tag, "seed": seed, "snr": snr_name(v), "method": d, "n": len(stems), **r}

    rows = [ev(args.seed, v, m, test, "C") for v in SNR_LEVELS for m in METHODS]
    rows += [ev(args.seed, v, m + "_tunedinf", test, "A") for v in SNR_LEVELS[1:] for m in METHODS[1:]]
    var_stems = test[: man.get("var_n", 0)]
    seeds = [args.seed] + man.get("var_seeds", [])
    rows += [ev(sd, v, m, var_stems, "var") for v in man.get("var_snr_db", []) for m in METHODS for sd in seeds] \
        if var_stems else []
    keys = ["set", "seed", "snr", "method", "n", "precision", "recall", "f1", "map50"]
    with open(os.path.join(run_dir(args), "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def cell(r):
        return f"{r['precision']:.3f} / {r['recall']:.3f} / {r['map50']:.3f}"

    def get(tag, v, m):
        return next(r for r in rows if r["set"] == tag and r["snr"] == snr_name(v) and r["method"] == m)

    n = get("C", SNR_LEVELS[0], "csa")["n"]
    lines = [f"Base seed {args.seed}, {n} test scenes, no speckle. Cells: P / R / mAP@0.5", "",
             "C: parameters tuned per SNR level", "",
             "| SNR (dB) | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for v in SNR_LEVELS:
        lines.append(f"| {v:g} | " + " | ".join(cell(get("C", v, m)) for m in METHODS) + " |")
    lines += ["", "A: parameters tuned at SNR = inf, applied at every level (CSA has no parameters)", "",
              "| SNR (dB) | " + " | ".join(METHODS[1:]) + " |", "|---|" + "---|" * (len(METHODS) - 1)]
    for v in SNR_LEVELS:
        cells = [cell(get("C", v, m)) if v == float("inf") else cell(get("A", v, m + "_tunedinf")) for m in METHODS[1:]]
        lines.append(f"| {v:g} | " + " | ".join(cells) + " |")
    if var_stems:
        lines += ["", f"Noise-realization spread: {len(var_stems)} scenes x {len(seeds)} seeds, C parameters", "",
                  "| SNR (dB) | method | mAP@0.5 mean | std | min | max |", "|---|---|---|---|---|---|"]
        for v in man["var_snr_db"]:
            for m in METHODS:
                mp = np.array([r["map50"] for r in rows if r["set"] == "var" and r["snr"] == snr_name(v)
                               and r["method"] == m])
                lines.append(f"| {v:g} | {m} | {mp.mean():.4f} | {mp.std(ddof=1):.4f} | {mp.min():.4f} | {mp.max():.4f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "results.md"), "w") as f:
        f.write(text)
    print(text)


def cmd_iters(args):
    """Iteration counts of L1/2 & TV on a sample of test scenes with the parameters actually used
    (C: per level; A: tuned at SNR = inf), no JPG / YOLO: iterations_sample.csv + iterations.md."""
    import cupy as cp
    from .refine import apply
    ctx = Context(cp)
    params_c, params_a = load_params(args)
    stems = iteration_sample(load_split(args)["test"], args.n_sample)
    path = os.path.join(run_dir(args), "iterations_sample.csv")
    if os.path.exists(path):
        os.remove(path)
    log = IterLog(path, "sample")
    for i, stem in enumerate(stems):
        amp = ctx.load_amp(stem)
        crop = clean_crop(ctx, amp)
        p = scene_power(cp, crop)
        w = cp.asarray(noise_field(stem, crop.shape, args.seed))
        for v in SNR_LEVELS:
            h = noisy(cp, crop, w, v, p)
            jobs = [("C", params_c[snr_name(v)]["l12tv"])] + ([("A", params_a["l12tv"])] if v != float("inf") else [])
            for tag, prm in jobs:
                t0 = time.perf_counter()
                _, info = apply(cp, "l12tv", h, prm, MARGIN)
                cp.cuda.Stream.null.synchronize()
                log.log(stem, args.seed, snr_name(v), "l12tv", f"{tag}:{config_name('l12tv', prm)}", info,
                        time.perf_counter() - t0)
        del crop, w
        cp.get_default_memory_pool().free_all_blocks()
        if (i + 1) % 20 == 0:
            print(f"  iters [{i + 1}/{len(stems)}]", flush=True)
    log.f.close()
    text = summarize_iterations(path, os.path.join(run_dir(args), "iterations.md"), group=("level", "config"))
    print(f"{len(stems)} test scenes (random sample, seed 0), tolerance 1e-4, cap 300\n" + text)


def cmd_report(args):
    rows = list(csv.DictReader(open(os.path.join(run_dir(args), "timing.csv"))))
    lines = ["| phase | step | calls | mean (s) | total (h) |", "|---|---|---|---|---|"]
    for phase in ("tune", "run", "eval"):
        for st in sorted({r["step"] for r in rows if r["phase"] == phase}):
            v = [float(r["seconds"]) for r in rows if r["phase"] == phase and r["step"] == st]
            lines.append(f"| {phase} | {st} | {len(v)} | {np.mean(v):.3f} | {np.sum(v) / 3600:.2f} |")
    total = sum(float(r["seconds"]) for r in rows)
    lines.append(f"| **total** | | | | **{total / 3600:.2f}** |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "timing_report.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "validate", "tune", "run", "eval", "report", "iters"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0, help="base noise seed")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-tune", type=int, default=100)
    ap.add_argument("--n", type=int, default=None, help="only the first N scenes of the split")
    ap.add_argument("--max-rounds", type=int, default=4, help="tune: grid-extension rounds")
    ap.add_argument("--var-n", type=int, default=0, help="first N test scenes also get --var-seeds")
    ap.add_argument("--var-seeds", default="1,2,3,4")
    ap.add_argument("--var-snr", default="5,0")
    ap.add_argument("--n-sample", type=int, default=100, help="iters: test scenes sampled")
    ap.add_argument("--exclude-cpus", default="1,7")
    args = ap.parse_args()
    exclude_cpus(args.exclude_cpus)
    {"split": cmd_split, "validate": cmd_validate, "tune": cmd_tune, "run": cmd_run, "eval": cmd_eval,
     "report": cmd_report, "iters": cmd_iters}[args.cmd](args)


if __name__ == "__main__":
    main()
