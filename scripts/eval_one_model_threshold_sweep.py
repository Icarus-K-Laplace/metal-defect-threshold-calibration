import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def infer_label_dir(img_dir: Path) -> Path:
    s = img_dir.as_posix()
    if "/images/" in s:
        return Path(s.replace("/images/", "/labels/"))
    if s.endswith("/images"):
        return Path(s[:-7] + "/labels")
    return img_dir.parent / "labels" / img_dir.name

def list_images(img_dir: Path):
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def load_yolo_gt(label_path: Path, img_w: int, img_h: int):
    gts = []
    if not label_path.exists():
        return gts
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                gts.append((cls_id, x1, y1, x2, y2))
            except Exception:
                continue
    return gts

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def evaluate_detector(model: YOLO, img_dir: Path, label_dir: Path, conf_thr: float,
                      iou_thr: float = 0.5, nms_iou: float = 0.7, device=None):
    image_paths = list_images(img_dir)
    TP = FP = FN = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_path = label_dir / f"{img_path.stem}.txt"
        gts = load_yolo_gt(gt_path, w, h)
        matched_gt = [False] * len(gts)

        result = model.predict(
            source=str(img_path),
            conf=conf_thr,
            iou=nms_iou,
            device=device,
            verbose=False
        )[0]

        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append((cls_id, score, x1, y1, x2, y2))

        preds = sorted(preds, key=lambda x: x[1], reverse=True)

        for p_cls, p_score, px1, py1, px2, py2 in preds:
            best_iou = 0.0
            best_idx = -1
            for gi, gt in enumerate(gts):
                if matched_gt[gi]:
                    continue
                gt_cls, gx1, gy1, gx2, gy2 = gt
                if gt_cls != p_cls:
                    continue
                cur_iou = iou_xyxy((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = gi

            if best_idx >= 0 and best_iou >= iou_thr:
                TP += 1
                matched_gt[best_idx] = True
            else:
                FP += 1

        FN += sum(1 for m in matched_gt if not m)

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"TP": TP, "FP": FP, "FN": FN, "Precision": precision, "Recall": recall, "F1": f1}

def sweep_thresholds(model: YOLO, img_dir: Path, label_dir: Path, conf_grid,
                     iou_thr=0.5, nms_iou=0.7, device=None):
    rows = []
    best_row = None

    for conf_thr in conf_grid:
        metrics = evaluate_detector(
            model=model, img_dir=img_dir, label_dir=label_dir,
            conf_thr=float(conf_thr), iou_thr=iou_thr, nms_iou=nms_iou, device=device
        )
        row = {"conf_thr": float(conf_thr), **metrics}
        rows.append(row)
        if best_row is None or row["F1"] > best_row["F1"]:
            best_row = row

    return pd.DataFrame(rows), best_row

def parse_data_yaml(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base_dir = data.get("path", None)

    def resolve(p):
        pp = Path(p)
        if pp.is_absolute():
            return pp
        if base_dir is None:
            return pp
        return Path(base_dir) / pp

    train_img = resolve(data["train"])
    val_img = resolve(data["val"])
    test_img = resolve(data.get("test", data["val"]))

    return {
        "train_img": train_img,
        "val_img": val_img,
        "test_img": test_img,
        "train_lbl": infer_label_dir(train_img),
        "val_lbl": infer_label_dir(val_img),
        "test_lbl": infer_label_dir(test_img),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--conf-grid", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    args = parser.parse_args()

    paths = parse_data_yaml(args.data)
    val_img, val_lbl = paths["val_img"], paths["val_lbl"]
    test_img, test_lbl = paths["test_img"], paths["test_lbl"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    sweep_df, best_row = sweep_thresholds(
        model=model,
        img_dir=val_img,
        label_dir=val_lbl,
        conf_grid=args.conf_grid,
        iou_thr=args.iou_thr,
        nms_iou=args.nms_iou,
        device=args.device
    )
    sweep_df.to_csv(outdir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")

    best_conf = float(best_row["conf_thr"])
    test_metrics = evaluate_detector(
        model=model,
        img_dir=test_img,
        label_dir=test_lbl,
        conf_thr=best_conf,
        iou_thr=args.iou_thr,
        nms_iou=args.nms_iou,
        device=args.device
    )

    result = {
        "weights": args.weights,
        "best_conf": best_conf,
        "val_TP": int(best_row["TP"]),
        "val_FP": int(best_row["FP"]),
        "val_FN": int(best_row["FN"]),
        "val_Precision": float(best_row["Precision"]),
        "val_Recall": float(best_row["Recall"]),
        "val_F1": float(best_row["F1"]),
        "test_TP": test_metrics["TP"],
        "test_FP": test_metrics["FP"],
        "test_FN": test_metrics["FN"],
        "test_Precision": float(test_metrics["Precision"]),
        "test_Recall": float(test_metrics["Recall"]),
        "test_F1": float(test_metrics["F1"]),
    }

    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    pd.DataFrame([result]).to_csv(outdir / "result.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
