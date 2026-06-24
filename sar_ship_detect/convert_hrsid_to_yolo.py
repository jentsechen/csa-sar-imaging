#!/usr/bin/env python3
"""Convert HRSID COCO annotations -> YOLO format (single class: ship).

Builds HRSID_YOLO/ with the ultralytics-standard layout:

    HRSID_YOLO/
      images/{train,val}/   symlinks to HRSID_JPG/JPEGImages (no data duplicated)
      labels/{train,val}/   YOLO .txt (class 0 = ship)
      data.yaml             ready for `yolo train data=.../data.yaml`

HRSID's "test2017" split is used as the YOLO `val` set (standard practice for
this dataset). Run from the sar-ship-detect/ directory.
"""
import argparse
import json
import os

SPLITS = {"train": "train2017.json", "val": "test2017.json"}


def convert_split(coco_json, img_src_dir, out_root, split):
    data = json.load(open(coco_json))
    images = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_out = os.path.join(out_root, "images", split)
    lbl_out = os.path.join(out_root, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_img = n_box = 0
    for img_id, im in images.items():
        base = os.path.basename(im["file_name"])
        stem, _ = os.path.splitext(base)
        W, H = im["width"], im["height"]

        # symlink the image (relative) instead of copying 1.2 GB
        src = os.path.join(img_src_dir, base)
        dst = os.path.join(img_out, base)
        if not os.path.exists(dst):
            os.symlink(os.path.relpath(src, img_out), dst)

        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]  # COCO: top-left x,y + w,h (pixels)
            xc, yc = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
            n_box += 1
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n_img += 1
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="HRSID_JPG/annotations")
    ap.add_argument("--img-dir", default="HRSID_JPG/JPEGImages")
    ap.add_argument("--out", default="HRSID_YOLO")
    args = ap.parse_args()

    img_dir_abs = os.path.abspath(args.img_dir)
    for split, jf in SPLITS.items():
        ni, nb = convert_split(os.path.join(args.ann_dir, jf), img_dir_abs, args.out, split)
        print(f"{split:5s}: {ni} images, {nb} ship boxes")

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"# HRSID — SAR ship detection (single class), COCO->YOLO converted\n"
            f"path: {os.path.abspath(args.out)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names:\n  0: ship\n"
        )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
