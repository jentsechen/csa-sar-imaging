#!/usr/bin/env python3
"""Compare YOLO Precision/Recall/mAP@0.5 across three versions of the same
scenes, split by inshore vs offshore, to separate thresholding loss from
CSA-reconstruction loss and see whether either effect differs by scene type:

    original  - raw HRSID JPG, no processing
    threshold - point_target_location/*.json (threshold=150) rendered directly
                to JPG, no echo/CSA involved
    csa       - threshold -> echo signal -> CSA focus -> mag/dB -> JPG

Restricted to scenes that have made it all the way through to csa_jpg/, for a
fair comparison. Inshore/offshore membership comes from the official HRSID
annotation split (sar_ship_detect/HRSID_JPG/inshore_offshore/*.json).

Usage:
    python compare_original_vs_csa.py --device cpu
"""
import argparse
import json
import os

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_IMAGES_DIR = os.path.join(BASE, "images")
THRESHOLD_JPG_DIR = os.path.join(BASE, "threshold_jpg")
CSA_JPG_DIR = os.path.join(BASE, "csa_jpg")
LABELS_DIR = os.path.join(BASE, "labels")
WEIGHTS = os.path.join(BASE, "..", "sar_ship_detect", "weights", "best.pt")
INSHORE_OFFSHORE_DIR = os.path.join(BASE, "..", "sar_ship_detect", "HRSID_JPG", "inshore_offshore")

# name -> source image dir
SETS = {
    "original": ORIGINAL_IMAGES_DIR,
    "threshold": THRESHOLD_JPG_DIR,
    "csa": CSA_JPG_DIR,
}


def processed_stems():
    """Scenes that have made it all the way through to csa_jpg/."""
    return sorted(os.path.splitext(f)[0] for f in os.listdir(CSA_JPG_DIR) if f.endswith(".jpg"))


def split_by_region(stems):
    with open(os.path.join(INSHORE_OFFSHORE_DIR, "inshore.json")) as f:
        inshore_names = {os.path.basename(im["file_name"]) for im in json.load(f)["images"]}
    with open(os.path.join(INSHORE_OFFSHORE_DIR, "offshore.json")) as f:
        offshore_names = {os.path.basename(im["file_name"]) for im in json.load(f)["images"]}

    regions = {"inshore": [], "offshore": []}
    for stem in stems:
        name = stem + ".jpg"
        if name in inshore_names:
            regions["inshore"].append(stem)
        elif name in offshore_names:
            regions["offshore"].append(stem)
    return regions


def build_eval_set(image_dir, eval_dir, stems):
    images_out = os.path.join(eval_dir, "images")
    labels_out = os.path.join(eval_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for stem in stems:
        img_link = os.path.join(images_out, stem + ".jpg")
        if not os.path.exists(img_link):
            os.symlink(os.path.join(image_dir, stem + ".jpg"), img_link)

        lbl_src = os.path.join(LABELS_DIR, stem + ".txt")
        lbl_link = os.path.join(labels_out, stem + ".txt")
        if os.path.exists(lbl_src) and not os.path.exists(lbl_link):
            os.symlink(lbl_src, lbl_link)

    yaml_path = os.path.join(eval_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {eval_dir}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: ship\n"
        )
    return yaml_path


def run_val(model, yaml_path, eval_dir, args):
    metrics = model.val(
        data=yaml_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=eval_dir,
        name="run",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    return box.mp, box.mr, box.map50, box.map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    stems = processed_stems()
    regions = split_by_region(stems)
    print(f"Total: {len(stems)} scenes -> inshore: {len(regions['inshore'])}, "
          f"offshore: {len(regions['offshore'])}\n")

    model = YOLO(WEIGHTS)

    all_results = {}
    for region, region_stems in regions.items():
        if not region_stems:
            continue
        for name, image_dir in SETS.items():
            eval_dir = os.path.join(BASE, f"{region}_{name}_eval")
            yaml_path = build_eval_set(image_dir, eval_dir, region_stems)
            all_results[(region, name)] = run_val(model, yaml_path, eval_dir, args)

    for region, region_stems in regions.items():
        if not region_stems:
            continue
        print(f"\n=== {region} (n={len(region_stems)}) ===")
        print(f"{'Set':<10} {'Precision':>10} {'Recall':>8} {'mAP@0.5':>9} {'mAP@0.5:0.95':>13}")
        for name in SETS:
            p, r, map50, map5095 = all_results[(region, name)]
            print(f"{name:<10} {p:>10.4f} {r:>8.4f} {map50:>9.4f} {map5095:>13.4f}")


if __name__ == "__main__":
    main()
