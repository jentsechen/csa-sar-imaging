"""YOLO Precision / Recall / mAP@0.5 of a JPG directory against the HRSID labels,
with the same settings as end_to_end_pipeline/eval_union_csa.py."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_DIR = os.path.join(REPO, "end_to_end_pipeline", "labels")
WEIGHTS = os.path.join(REPO, "sar_ship_detect", "weights", "best.pt")


def load_model():
    from ultralytics import YOLO
    return YOLO(WEIGHTS)


def evaluate(model, image_dir, stems, work_dir, imgsz=800, conf=0.25, iou=0.45, device="0"):
    """Symlinks <image_dir>/<stem>.jpg and its label into work_dir and runs model.val.
    Returns dict(precision, recall, map50, f1)."""
    images_out = os.path.join(work_dir, "images")
    labels_out = os.path.join(work_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    wanted = set(stems)
    for d in (images_out, labels_out):
        for f in os.listdir(d):
            if os.path.splitext(f)[0] not in wanted:
                os.remove(os.path.join(d, f))
    cache = os.path.join(work_dir, "labels.cache")
    if os.path.exists(cache):
        os.remove(cache)
    for stem in stems:
        for src, dst in ((os.path.join(image_dir, stem + ".jpg"), os.path.join(images_out, stem + ".jpg")),
                         (os.path.join(LABELS_DIR, stem + ".txt"), os.path.join(labels_out, stem + ".txt"))):
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)
    yaml_path = os.path.join(work_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {work_dir}\ntrain: images\nval: images\nnc: 1\nnames:\n  0: ship\n")
    m = model.val(data=yaml_path, imgsz=imgsz, conf=conf, iou=iou, device=device, project=work_dir,
                  name="run", exist_ok=True, plots=False, verbose=False, workers=0)
    p, r, map50 = float(m.box.mp), float(m.box.mr), float(m.box.map50)
    return {"precision": p, "recall": r, "map50": map50, "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0}
