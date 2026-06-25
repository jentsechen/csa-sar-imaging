#!/usr/bin/env python3
"""Create inshore / offshore val subsets from the HRSID inshore_offshore annotations.

Reads HRSID_JPG/inshore_offshore/{inshore,offshore}.json and the existing val
split (test2017.json), then builds two filtered symlink trees under HRSID_YOLO:

    images/val_inshore/   images/val_offshore/
    labels/val_inshore/   labels/val_offshore/

Also writes data_inshore.yaml and data_offshore.yaml alongside the existing
data.yaml so you can run val 3 ways without any changes to infer.py flags.

Run from the sar_ship_detect/ directory:
    python split_inshore_offshore.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
YOLO = os.path.join(BASE, "HRSID_YOLO")
ANN = os.path.join(BASE, "HRSID_JPG", "annotations")
INSHORE_OFFSHORE = os.path.join(BASE, "HRSID_JPG", "inshore_offshore")


def load_filenames(json_path):
    data = json.load(open(json_path))
    return {os.path.basename(im["file_name"]) for im in data["images"]}


def make_subset(val_files, subset_files, split_name):
    members = sorted(val_files & subset_files)

    img_src = os.path.join(YOLO, "images", "val")
    lbl_src = os.path.join(YOLO, "labels", "val")
    img_dst = os.path.join(YOLO, "images", split_name)
    lbl_dst = os.path.join(YOLO, "labels", split_name)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    n = 0
    for fname in members:
        stem = os.path.splitext(fname)[0]

        img_link = os.path.join(img_dst, fname)
        if not os.path.exists(img_link):
            os.symlink(
                os.path.relpath(os.path.join(img_src, fname), img_dst),
                img_link,
            )

        lbl_link = os.path.join(lbl_dst, stem + ".txt")
        lbl_file = os.path.join(lbl_src, stem + ".txt")
        if os.path.exists(lbl_file) and not os.path.exists(lbl_link):
            os.symlink(
                os.path.relpath(lbl_file, lbl_dst),
                lbl_link,
            )
        n += 1

    return n


def write_yaml(split_name, nc, names):
    path = os.path.join(YOLO, f"data_{split_name}.yaml")
    with open(path, "w") as f:
        f.write(
            f"# HRSID — {split_name} subset\n"
            f"path: {YOLO}\n"
            f"train: images/train\n"
            f"val: images/{split_name}\n"
            f"nc: {nc}\n"
            f"names:\n"
        )
        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")
    return path


def main():
    val_files = load_filenames(os.path.join(ANN, "test2017.json"))
    inshore_files = load_filenames(os.path.join(INSHORE_OFFSHORE, "inshore.json"))
    offshore_files = load_filenames(os.path.join(INSHORE_OFFSHORE, "offshore.json"))

    n_in = make_subset(val_files, inshore_files, "val_inshore")
    n_off = make_subset(val_files, offshore_files, "val_offshore")

    yaml_in = write_yaml("val_inshore", 1, ["ship"])
    yaml_off = write_yaml("val_offshore", 1, ["ship"])

    print(f"val_inshore  : {n_in:4d} images  -> {yaml_in}")
    print(f"val_offshore : {n_off:4d} images  -> {yaml_off}")
    print(f"combined     : {len(val_files):4d} images  -> HRSID_YOLO/data.yaml  (unchanged)")
    print()
    print("Run inference:")
    print("  python infer.py --val --data HRSID_YOLO/data_val_inshore.yaml")
    print("  python infer.py --val --data HRSID_YOLO/data_val_offshore.yaml")
    print("  python infer.py --val --data HRSID_YOLO/data.yaml")


if __name__ == "__main__":
    main()
