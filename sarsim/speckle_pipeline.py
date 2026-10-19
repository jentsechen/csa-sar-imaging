#!/usr/bin/env python3
"""Speckle-level sweep: unmasked scene + 1-look Rician speckle at several SSRs
(sarsim/speckle.py) -> exact echoes (sarsim.echo.gen_echo_signals, clean scene
and speckle realizations in one pass) -> CSA -> CSA / L0 / L1 / L1/2&TV
(sarsim/refine.py) -> 30 dB JPG -> YOLO Precision / Recall / mAP@0.5.

Steps (each writes under end_to_end_pipeline/speckle/<run>/):
    split     tune / test split of the 1593 HRSID scenes (seed 0, 100 tune scenes)
    validate  mixed-precision multi-set echo vs the FP64 kernel, and speckle statistics
    tune      parameter grid on the tune scenes at --tune-ssr; picks the parameters by
              tune-set mAP@0.5 (ties within 0.002: higher F1, then the simpler setting)
    run       every SSR level x method on the test scenes with the chosen parameters;
              the first --var-n test scenes also get the extra seeds --var-seeds at --var-ssr
    eval      YOLO on every (seed, SSR, method) set -> results.csv / results.md, and the
              speckle-realization spread over the --var-n scenes (all seeds) -> variance.md
    report    timing summary and extrapolation to the full run (timing_report.md)

Usage:
    python -m sarsim.speckle_pipeline split --run full
    python -m sarsim.speckle_pipeline tune --run full
    python -m sarsim.speckle_pipeline run --run full --var-n 100
    python -m sarsim.speckle_pipeline eval --run full
"""
import argparse
import csv
import datetime
import itertools
import json
import os
import time

import numpy as np

from .pipeline import exclude_cpus, git_commit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(REPO, "end_to_end_pipeline")
INPUT_PAR = os.path.join(PIPE, "union_pipeline", "input_par.json")
IMAGES = os.path.join(PIPE, "images")
METHODS = ["csa", "l0", "l1", "l12tv"]
SSR_LEVELS = [float("inf"), 10.0, 5.0, 0.0, float("-inf")]
MARGIN = 112                # crop margin for TV: (800 + 2 * 112)^2 = 1024^2
ALPHAS_THRESHOLD = (0.0316, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40)
TUNE_GRID = {
    "l0": [{"alpha": a} for a in ALPHAS_THRESHOLD],
    "l1": [{"alpha": a} for a in ALPHAS_THRESHOLD],
    # first grid, then an extension toward larger alpha / smaller xi after the first
    # tuning picked alpha = 0.16 and xi = 1, both on the edge of the first grid
    "l12tv": [{"alpha": a, "xi": x, "tv": t}
              for a, x, t in itertools.product((0.01, 0.02, 0.0316, 0.05, 0.08, 0.16),
                                               (1.0, 10.0, 30.0), (0.001, 0.003, 0.01, 0.03))]
             + [{"alpha": a, "xi": x, "tv": t}
                for a, x, t in itertools.product((0.13, 0.16, 0.20, 0.25, 0.30),
                                                 (0.3, 0.5, 1.0, 2.0, 3.0), (0.01, 0.03, 0.1))
                if not (a == 0.16 and x == 1.0 and t in (0.01, 0.03))],
}


def ssr_name(v):
    return "ssr" + ("inf" if v == float("inf") else "-inf" if v == float("-inf") else f"{v:g}")


def config_name(method, params):
    return method + "".join(f"_{k}{v:g}" for k, v in params.items())


class Timer:
    def __init__(self, path, phase):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        self.phase = phase
        if new:
            self.w.writerow(["phase", "stem", "step", "seconds"])

    def log(self, stem, step, seconds):
        self.w.writerow([self.phase, stem, step, f"{seconds:.4f}"])
        self.f.flush()


class IterLog:
    """Per-call iteration record of the iterative refinement (L1/2 & TV): iterations.csv."""

    FIELDS = ["phase", "stem", "seed", "level", "method", "config", "n_iter", "final_rel", "converged", "seconds"]

    def __init__(self, path, phase):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        self.phase = phase
        if new:
            self.w.writerow(self.FIELDS)

    def log(self, stem, seed, level, method, config, info, seconds):
        if method != "l12tv":
            return
        self.w.writerow([self.phase, stem, seed, level, method, config, info["n_iter"], f"{info['rel']:.3e}",
                         int(info["converged"]), f"{seconds:.4f}"])
        self.f.flush()


def summarize_iterations(csv_path, md_path, group=("phase", "level", "config")):
    """Iteration-count distribution per group -> markdown table."""
    rows = list(csv.DictReader(open(csv_path)))
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[g] for g in group), []).append(r)
    lines = ["| " + " | ".join(group) + " | calls | mean | median | p95 | min | max | hit cap | s/call |",
             "|" + "---|" * (len(group) + 8)]
    for key in sorted(groups):
        v = groups[key]
        n = np.array([int(r["n_iter"]) for r in v])
        cap = sum(1 for r in v if r["converged"] == "0")
        sec = np.mean([float(r["seconds"]) for r in v])
        lines.append("| " + " | ".join(key) + f" | {len(n)} | {n.mean():.1f} | {np.median(n):.0f} | "
                     f"{np.percentile(n, 95):.0f} | {n.min()} | {n.max()} | {cap} | {sec:.3f} |")
    text = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(text)
    return text


def iteration_sample(stems, n, seed=0):
    """Deterministic random sample of n scenes (sorted)."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(stems, min(n, len(stems)), replace=False).tolist())


class Context:
    """GPU state shared by every scene: imaging axes and CSA."""

    def __init__(self, cp):
        from .imaging.csa import ChirpScalingAlgo
        from .params import build_imaging_axes, load_input_par
        self.cp = cp
        self.par = load_input_par(INPUT_PAR)
        self.ax = build_imaging_axes(self.par)
        self.csa = ChirpScalingAlgo(cp, self.ax, self.par)
        self.r0, self.c0 = self.ax["n_row"] * 3 // 8, self.ax["n_col"] * 3 // 8

    def load_amp(self, stem):
        import cv2
        img = cv2.imread(os.path.join(IMAGES, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(stem)
        return img

    def images(self, stem, amp, seeds):
        """Cropped CSA images (scene + MARGIN): the clean one and one speckle image per seed."""
        from .echo import gen_echo_signals
        from .params import build_point_target_list
        from .speckle import coef_sets, speckle_field
        fields = [speckle_field(stem, amp.shape, sd) for sd in seeds]
        az, rg, _ = build_point_target_list(amp, self.ax["n_row"], self.ax["n_col"],
                                            self.ax["pulse_rep_freq_hz"], self.ax["sampling_freq_hz"])
        echoes = gen_echo_signals(self.ax, az, rg, coef_sets(amp, fields))
        crops = []
        for q in range(echoes.shape[0]):
            crops.append(self.crop(self.csa.focus(echoes[q])).copy())
        del echoes
        return crops[0], dict(zip(seeds, crops[1:])), dict(zip(seeds, fields))

    def crop(self, h):
        """Scene crop (same rows / cols as sarsim.postproc.scene_crop) plus MARGIN on every side."""
        return h[self.r0 - MARGIN:self.r0 + 800 + MARGIN, self.c0 - MARGIN:self.c0 + 800 + MARGIN]


def mixed(h0, hs, ssr):
    from .speckle import ssr_weights
    c, d = ssr_weights(ssr)
    return h0 if d == 0 else (c * h0 + d * hs)


def to_gray(cp, scene_crop):
    """Complex scene crop -> uint8 30 dB JPG image, flipped up-down (as sarsim.postproc.process)."""
    from .postproc import db_to_gray
    power = cp.asnumpy(scene_crop.real ** 2 + scene_crop.imag ** 2)
    with np.errstate(divide="ignore"):
        mag_db = 10.0 * np.log10(power)
    return np.flipud(db_to_gray(mag_db))


def run_dir(args):
    return os.path.join(PIPE, "speckle", args.run)


def load_split(args):
    with open(os.path.join(run_dir(args), "split.json")) as f:
        return json.load(f)


def write_manifest(args, extra):
    path = os.path.join(run_dir(args), "manifest.json")
    m = {}
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
    m.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    m.update({"git_commit": git_commit(), "numpy": np.__version__, "base_seed": args.seed,
              "ssr_levels_db": [str(v) for v in SSR_LEVELS], "margin": MARGIN,
              "echo": "sarsim.echo.gen_echo_signals (FP64 geometry, FP32 chirp loop)", **extra})
    with open(path, "w") as f:
        json.dump(m, f, indent=2)


def cmd_split(args):
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    rng = np.random.default_rng(args.split_seed)
    tune = sorted(rng.choice(stems, args.n_tune, replace=False).tolist())
    test = sorted(set(stems) - set(tune))
    os.makedirs(run_dir(args), exist_ok=True)
    with open(os.path.join(run_dir(args), "split.json"), "w") as f:
        json.dump({"seed": args.split_seed, "tune": tune, "test": test}, f, indent=1)
    print(f"{len(tune)} tune / {len(test)} test scenes -> {run_dir(args)}/split.json")


def cmd_validate(args):
    import cupy as cp
    from .echo import gen_echo_signal
    from .params import build_point_target_list
    from .speckle import field_stats
    ctx = Context(cp)
    stem = args.stem or load_split(args)["test"][0]
    amp = ctx.load_amp(stem)
    sync = cp.cuda.Stream.null.synchronize
    t0 = time.perf_counter()
    h0, hs, fields = ctx.images(stem, amp, [args.seed])
    sync()
    t_multi = time.perf_counter() - t0
    hs, s = hs[args.seed], fields[args.seed]
    az, rg, coef = build_point_target_list(amp, ctx.ax["n_row"], ctx.ax["n_col"],
                                           ctx.ax["pulse_rep_freq_hz"], ctx.ax["sampling_freq_hz"])
    t0 = time.perf_counter()
    ref0 = ctx.crop(ctx.csa.focus(gen_echo_signal(ctx.ax, az, rg, coef)[0]))
    c = coef * s[amp > 0]
    ref_s = ctx.crop(ctx.csa.focus(gen_echo_signal(ctx.ax, az, rg, c.real)[0])
                     + 1j * ctx.csa.focus(gen_echo_signal(ctx.ax, az, rg, c.imag)[0]))
    sync()
    t_ref = time.perf_counter() - t0
    err = lambda a, b: 20 * np.log10(float(cp.linalg.norm(a - b) / cp.linalg.norm(b)))
    lines = [f"scene {stem}, base seed {args.seed}",
             f"mixed-precision multi-set echo (clean + speckle, one pass) vs FP64 kernel, CSA crop: "
             f"clean {err(h0, ref0):.1f} dB, speckle {err(hs, ref_s):.1f} dB; "
             f"time {t_multi:.2f} s vs {t_ref:.1f} s (3 FP64 echoes)", "",
             "| SSR (dB) | mean | contrast | contrast (theory) |", "|---|---|---|---|"]
    for v in SSR_LEVELS:
        st = field_stats(amp, s, v)
        lines.append(f"| {v:g} | {st['mean']:.4f} | {st['contrast']:.4f} | {st['contrast_theory']:.4f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "validate.md"), "w") as f:
        f.write(text)
    print(text)


def cmd_tune(args):
    import cupy as cp
    import cv2
    from .refine import apply
    ctx = Context(cp)
    stems = load_split(args)["tune"][: args.n]
    out = os.path.join(run_dir(args), "tune")
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "tune")
    iters = IterLog(os.path.join(run_dir(args), "iterations.csv"), "tune")
    configs = [("csa", {})] + [(m, p) for m in ("l0", "l1", "l12tv") for p in TUNE_GRID[m]]
    for i, stem in enumerate(stems):
        if all(os.path.exists(os.path.join(out, config_name(m, p), stem + ".jpg")) for m, p in configs):
            continue  # resume: every config already written (results are deterministic)
        t0 = time.perf_counter()
        amp = ctx.load_amp(stem)
        h0, hs, _ = ctx.images(stem, amp, [args.seed])
        crop = mixed(h0, hs[args.seed], args.tune_ssr)
        del h0, hs
        cp.cuda.Stream.null.synchronize()
        timer.log(stem, "echo+csa", time.perf_counter() - t0)
        for method, params in configs:
            d = os.path.join(out, config_name(method, params))
            if os.path.exists(os.path.join(d, stem + ".jpg")):
                continue  # resume / grid extension: already computed (deterministic)
            t0 = time.perf_counter()
            img, info = apply(cp, method, crop, params, MARGIN)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, stem + ".jpg"), to_gray(cp, img))
            timer.log(stem, method, time.perf_counter() - t0)
            iters.log(stem, args.seed, ssr_name(args.tune_ssr), method, config_name(method, params), info,
                      time.perf_counter() - t0)
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  tune [{i + 1}/{len(stems)}] {stem}", flush=True)

    from .detect_eval import evaluate, load_model
    model = load_model()
    rows = []
    t0 = time.perf_counter()
    for method, params in configs:
        name = config_name(method, params)
        r = evaluate(model, os.path.join(out, name), stems, os.path.join(run_dir(args), "_eval", "tune", name))
        rows.append({"method": method, **params, **r, "config": name})
    timer.log("-", f"yolo_eval_{len(stems) * len(configs)}", time.perf_counter() - t0)
    keys = ["config", "method", "alpha", "xi", "tv", "precision", "recall", "f1", "map50"]
    with open(os.path.join(run_dir(args), "tune_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    chosen, edges = {}, {}
    for method in ("l0", "l1", "l12tv"):
        cand = [r for r in rows if r["method"] == method]
        best = max(r["map50"] for r in cand)
        near = [r for r in cand if r["map50"] >= best - 0.002]
        best_f1 = max(r["f1"] for r in near)
        near = [r for r in near if r["f1"] >= best_f1 - 1e-9]
        # simplest: smallest alpha, then weakest TV, then smallest xi
        pick = min(near, key=lambda r: (r.get("alpha", 0), r.get("tv", 0), r.get("xi", 0)))
        chosen[method] = {k: pick[k] for k in ("alpha", "xi", "tv") if k in pick}
        # flag parameters that sit on the edge of their grid
        for k, v in chosen[method].items():
            values = sorted({p[k] for p in TUNE_GRID[method]})
            if v in (values[0], values[-1]):
                edges.setdefault(method, []).append(f"{k}={v:g} (grid {values[0]:g}..{values[-1]:g})")
    with open(os.path.join(run_dir(args), "chosen_params.json"), "w") as f:
        json.dump({"tune_ssr_db": args.tune_ssr, "n_tune": len(stems), "params": chosen,
                   "on_grid_edge": edges}, f, indent=2)
    write_manifest(args, {"tune_ssr_db": args.tune_ssr, "n_tune": len(stems), "chosen_params": chosen})
    print(json.dumps({"params": chosen, "on_grid_edge": edges}, indent=2))


def cmd_run(args):
    import cupy as cp
    import cv2
    from .refine import apply
    from .speckle import field_stats
    ctx = Context(cp)
    with open(os.path.join(run_dir(args), "chosen_params.json")) as f:
        params = {"csa": {}, **json.load(f)["params"]}
    stems = load_split(args)["test"][: args.n]
    var_stems = set(stems[: args.var_n])
    var_seeds = [int(v) for v in args.var_seeds.split(",") if v.strip()]
    var_ssr = [float(v) for v in args.var_ssr.split(",") if v.strip()]
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "run")
    iters = IterLog(os.path.join(run_dir(args), "iterations.csv"), "run")
    write_manifest(args, {"n_test": len(stems), "var_n": len(var_stems), "var_seeds": var_seeds,
                          "var_ssr_db": var_ssr, "params": params})
    stats_path = os.path.join(run_dir(args), "speckle_stats.csv")
    new = not os.path.exists(stats_path)
    stats_f = open(stats_path, "a", newline="")
    sw = csv.writer(stats_f)
    if new:
        sw.writerow(["stem", "seed", "ssr_db", "mean", "contrast", "contrast_theory"])
    def outputs(stem):
        seeds = [args.seed] + (var_seeds if stem in var_stems else [])
        return [os.path.join(run_dir(args), "test", f"seed{sd}", ssr_name(v), m, stem + ".jpg")
                for sd in seeds for v in (SSR_LEVELS if sd == args.seed else var_ssr) for m in METHODS]

    for i, stem in enumerate(stems):
        if all(os.path.exists(f) for f in outputs(stem)):
            continue  # resume: every output already written (results are deterministic)
        t0 = time.perf_counter()
        amp = ctx.load_amp(stem)
        seeds = [args.seed] + (var_seeds if stem in var_stems else [])
        h0, hs, fields = ctx.images(stem, amp, seeds)
        cp.cuda.Stream.null.synchronize()
        timer.log(stem, f"echo+csa_{len(seeds) + 1}sets", time.perf_counter() - t0)
        for seed in seeds:
            levels = SSR_LEVELS if seed == args.seed else var_ssr
            for ssr in levels:
                st = field_stats(amp, fields[seed], ssr)
                sw.writerow([stem, seed, ssr, f"{st['mean']:.5f}", f"{st['contrast']:.5f}",
                             f"{st['contrast_theory']:.5f}"])
                crop = mixed(h0, hs[seed], ssr)
                for method in METHODS:
                    t0 = time.perf_counter()
                    img, info = apply(cp, method, crop, params[method], MARGIN)
                    d = os.path.join(run_dir(args), "test", f"seed{seed}", ssr_name(ssr), method)
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, stem + ".jpg"), to_gray(cp, img))
                    timer.log(stem, method, time.perf_counter() - t0)
                    iters.log(stem, seed, ssr_name(ssr), method, config_name(method, params[method]), info,
                              time.perf_counter() - t0)
        stats_f.flush()
        del h0, hs
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  run [{i + 1}/{len(stems)}] {stem}", flush=True)
    stats_f.close()


def cmd_eval(args):
    from .detect_eval import evaluate, load_model
    with open(os.path.join(run_dir(args), "manifest.json")) as f:
        man = json.load(f)
    test = load_split(args)["test"]
    base = os.path.join(run_dir(args), "test")
    timer = Timer(os.path.join(run_dir(args), "timing.csv"), "eval")
    model = load_model()
    main_seed = f"seed{args.seed}"

    def ev(seed_dir, ssr, method, stems, tag):
        d = os.path.join(base, seed_dir, ssr_name(ssr), method)
        stems = [s for s in stems if os.path.exists(os.path.join(d, s + ".jpg"))]
        t0 = time.perf_counter()
        r = evaluate(model, d, stems, os.path.join(run_dir(args), "_eval", tag, seed_dir, ssr_name(ssr), method))
        timer.log("-", f"yolo_eval_{len(stems)}", time.perf_counter() - t0)
        return {"set": tag, "seed": seed_dir, "ssr": ssr_name(ssr), "method": method, "n": len(stems), **r}

    # main table: base seed, every test scene
    rows = [ev(main_seed, v, m, test, "main") for v in SSR_LEVELS for m in METHODS]
    # speckle-realization spread: the var_n scenes, every seed (base seed included)
    var_stems = test[: man.get("var_n", 0)]
    seeds = [args.seed] + man.get("var_seeds", [])
    var_rows = [ev(f"seed{sd}", v, m, var_stems, "var") for v in man.get("var_ssr_db", [])
                for m in METHODS for sd in seeds] if var_stems else []
    keys = ["set", "seed", "ssr", "method", "n", "precision", "recall", "f1", "map50"]
    with open(os.path.join(run_dir(args), "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows + var_rows)
    lines = [f"Base seed {args.seed}, {rows[0]['n']} test scenes. Cells: P / R / mAP@0.5", "",
             "| SSR | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for v in SSR_LEVELS:
        cells = [next(f"{r['precision']:.3f} / {r['recall']:.3f} / {r['map50']:.3f}"
                      for r in rows if r["ssr"] == ssr_name(v) and r["method"] == m) for m in METHODS]
        lines.append(f"| {ssr_name(v)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "results.md"), "w") as f:
        f.write(text)
    print(text)
    if var_rows:
        vl = [f"{len(var_stems)} scenes x {len(seeds)} speckle seeds ({', '.join(map(str, seeds))}).", "",
              "| SSR | method | mAP@0.5 mean | std | min | max | recall std |", "|---|---|---|---|---|---|---|"]
        for v in man["var_ssr_db"]:
            for m in METHODS:
                rs = [r for r in var_rows if r["ssr"] == ssr_name(v) and r["method"] == m]
                mp = np.array([r["map50"] for r in rs])
                rc = np.array([r["recall"] for r in rs])
                vl.append(f"| {ssr_name(v)} | {m} | {mp.mean():.4f} | {mp.std(ddof=1):.4f} | {mp.min():.4f} | "
                          f"{mp.max():.4f} | {rc.std(ddof=1):.4f} |")
        vtext = "\n".join(vl) + "\n"
        with open(os.path.join(run_dir(args), "variance.md"), "w") as f:
            f.write(vtext)
        print(vtext)


def cmd_iters(args):
    """Iteration counts of L1/2 & TV with the chosen parameters on a sample of test scenes (no
    JPG / YOLO): iterations_sample.csv + iterations.md."""
    import cupy as cp
    from .refine import apply
    ctx = Context(cp)
    with open(os.path.join(run_dir(args), "chosen_params.json")) as f:
        prm = json.load(f)["params"]["l12tv"]
    stems = iteration_sample(load_split(args)["test"], args.n_sample)
    path = os.path.join(run_dir(args), "iterations_sample.csv")
    if os.path.exists(path):
        os.remove(path)
    log = IterLog(path, "sample")
    for i, stem in enumerate(stems):
        amp = ctx.load_amp(stem)
        h0, hs, _ = ctx.images(stem, amp, [args.seed])
        for v in SSR_LEVELS:
            t0 = time.perf_counter()
            _, info = apply(cp, "l12tv", mixed(h0, hs[args.seed], v), prm, MARGIN)
            cp.cuda.Stream.null.synchronize()
            log.log(stem, args.seed, ssr_name(v), "l12tv", config_name("l12tv", prm), info, time.perf_counter() - t0)
        del h0, hs
        cp.get_default_memory_pool().free_all_blocks()
        if (i + 1) % 20 == 0:
            print(f"  iters [{i + 1}/{len(stems)}]", flush=True)
    log.f.close()
    text = summarize_iterations(path, os.path.join(run_dir(args), "iterations.md"), group=("level", "config"))
    print(f"{len(stems)} test scenes (random sample, seed 0), tolerance 1e-4, cap 300\n" + text)


def cmd_report(args):
    """Mean seconds per step from timing.csv and an extrapolation to the full experiment."""
    rows = list(csv.DictReader(open(os.path.join(run_dir(args), "timing.csv"))))
    with open(os.path.join(run_dir(args), "manifest.json")) as f:
        man = json.load(f)
    split = load_split(args)
    mean = lambda phase, pred: np.mean([float(r["seconds"]) for r in rows
                                        if r["phase"] == phase and pred(r["step"])] or [np.nan])

    lines = ["| phase | step | calls | mean (s) | total (s) |", "|---|---|---|---|---|"]
    for phase in ("tune", "run", "eval"):
        for st in sorted({r["step"] for r in rows if r["phase"] == phase}):
            v = [float(r["seconds"]) for r in rows if r["phase"] == phase and r["step"] == st]
            lines.append(f"| {phase} | {st} | {len(v)} | {np.mean(v):.3f} | {np.sum(v):.1f} |")
    eval_rows = [r for r in rows if r["step"].startswith("yolo_eval")]
    eval_rate = sum(float(r["seconds"]) for r in eval_rows) / max(sum(int(r["step"].split("_")[-1]) for r in eval_rows), 1)

    n_tune, n_test, n_var, n_var_seeds = len(split["tune"]), len(split["test"]), 100, 4
    n_cfg = 1 + sum(len(v) for v in TUNE_GRID.values())
    echo_2 = mean("run", lambda s: s == "echo+csa_2sets")
    echo_var = mean("run", lambda s: s == f"echo+csa_{n_var_seeds + 2}sets")
    if np.isnan(echo_var):  # pilot used fewer seeds: scale the measured extra cost per set
        var_steps = [s for s in {r["step"] for r in rows} if s.startswith("echo+csa_") and s != "echo+csa_2sets"]
        if var_steps:
            k = int(var_steps[0].split("_")[1].rstrip("sets"))
            echo_var = echo_2 + (mean("run", lambda s: s == var_steps[0]) - echo_2) / (k - 2) * n_var_seeds
    per_method = {m: mean("run", lambda s, m=m: s == m) for m in METHODS}
    level_cost = sum(per_method.values())
    tune_methods = sum(mean("tune", lambda s, m=m: s == m) * (len(TUNE_GRID.get(m, [None])))
                       for m in METHODS)
    est = {
        f"tune: {n_tune} scenes x (echo + {n_cfg} configs)": n_tune * (mean("tune", lambda s: s == "echo+csa") + tune_methods),
        "tune: YOLO eval": n_tune * n_cfg * eval_rate,
        f"run: {n_test} scenes x (echo + 5 SSR x 4 methods)": n_test * (echo_2 + len(SSR_LEVELS) * level_cost),
        f"run: variance extra, {n_var} scenes x {n_var_seeds} seeds x 2 SSR": n_var * ((echo_var - echo_2) + n_var_seeds * 2 * level_cost),
        "eval: YOLO": (n_test * len(SSR_LEVELS) + n_var * (n_var_seeds + 1) * 2) * len(METHODS) * eval_rate,
    }
    lines += ["", f"YOLO eval: {eval_rate * 1e3:.1f} ms per image (incl. per-set overhead)", "",
              "| full-run part | estimate (h) |", "|---|---|"]
    lines += [f"| {k} | {v / 3600:.2f} |" for k, v in est.items()]
    lines.append(f"| **total** | **{np.nansum(list(est.values())) / 3600:.2f}** |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir(args), "timing_report.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "validate", "tune", "run", "eval", "report", "iters"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0, help="base speckle seed")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-tune", type=int, default=100)
    ap.add_argument("--n", type=int, default=None, help="only the first N scenes of the split")
    ap.add_argument("--tune-ssr", type=float, default=5.0)
    ap.add_argument("--var-n", type=int, default=0, help="first N test scenes also get --var-seeds")
    ap.add_argument("--var-seeds", default="1,2,3,4")
    ap.add_argument("--var-ssr", default="5,0")
    ap.add_argument("--stem", default=None, help="validate: scene (default: first test scene)")
    ap.add_argument("--n-sample", type=int, default=100, help="iters: test scenes sampled")
    ap.add_argument("--exclude-cpus", default="1,7")
    args = ap.parse_args()
    exclude_cpus(args.exclude_cpus)
    {"split": cmd_split, "validate": cmd_validate, "tune": cmd_tune, "run": cmd_run, "eval": cmd_eval,
     "report": cmd_report, "iters": cmd_iters}[args.cmd](args)


if __name__ == "__main__":
    main()
