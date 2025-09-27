#!/usr/bin/env python3
import json
import argparse
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    f1_score,
    balanced_accuracy_score,
)

CLASS_NAMES = ["C1", "C2", "C3"]

def load_data(pred_path, gt_path):
    # Load JSONs
    with open(pred_path, 'r') as f:
        preds = json.load(f)["images"]
    with open(gt_path, 'r') as f:
        gts   = json.load(f)["images"]

    # Build maps by file_name
    gt_map     = { img["file_name"]: img["ds"]               for img in gts }
    uncert_map = { img["file_name"]: img["uncertainty_ds"]   for img in gts }

    y_prob, y_pred, y_true, y_uncert = [], [], [], []
    for img in preds:
        fn = img["file_name"]
        if fn not in gt_map:
            raise KeyError(f"No GT 'ds' for {fn}")
        if fn not in uncert_map:
            raise KeyError(f"No GT 'uncertainty_ds' for {fn}")

        y_prob.append(   img["pred_ds_prob"] )
        y_pred.append(   img["pred_ds"]      )
        y_true.append(   gt_map[fn]          )
        y_uncert.append( uncert_map[fn]      )

    return (
        np.array(y_prob,    dtype=float),
        np.array(y_pred,    dtype=int),
        np.array(y_true,    dtype=float),
        np.array(y_uncert,  dtype=float),
    )

def compute_primary(y_prob, y_true, y_uncert):
    print("\nPRIMARY METRICS\n" + "-"*30)
    # 1) mAP per class + overall (still vs. true ds)
    aps = []
    print("mAP (average precision) [vs. ds]:")
    for i, name in enumerate(CLASS_NAMES):
        ap = average_precision_score(y_true[:,i], y_prob[:,i])
        aps.append(ap)
        print(f"  {name}: {ap:.4f}")
    print(f"  overall: {np.mean(aps):.4f}")

    # 2) Brier score (MSE) per class + average (vs. uncertainty_ds)
    bs = []
    print("\nBrier score (MSE) [vs. uncertainty_ds]:")
    for i, name in enumerate(CLASS_NAMES):
        msec = mean_squared_error(y_uncert[:,i], y_prob[:,i])
        bs.append(msec)
        print(f"  {name}: {msec:.4f}")
    print(f"  overall: {np.mean(bs):.4f}")

def compute_secondary(y_pred, y_true):
    print("\nSECONDARY METRICS\n" + "-"*30)
    # 3) F1 score per class + average
    f1s = []
    print("F1 score [vs. ds]:")
    for i, name in enumerate(CLASS_NAMES):
        score = f1_score(y_true[:,i], y_pred[:,i])
        f1s.append(score)
        print(f"  {name}: {score:.4f}")
    print(f"  overall: {np.mean(f1s):.4f}")

    # 4) Balanced accuracy per class + average
    bas = []
    print("\nBalanced accuracy [vs. ds]:")
    for i, name in enumerate(CLASS_NAMES):
        bal = balanced_accuracy_score(y_true[:,i], y_pred[:,i])
        bas.append(bal)
        print(f"  {name}: {bal:.4f}")
    print(f"  overall: {np.mean(bas):.4f}")

def main():
    p = argparse.ArgumentParser(
        description="Compute multi-label classification metrics",
    )
    p.add_argument("--pred", required=True,
                   help="Predictions JSON")
    p.add_argument("--gt",   required=True,
                   help="GT JSON")
    args = p.parse_args()

    y_prob, y_pred, y_true, y_uncert = load_data(args.pred, args.gt)
    compute_primary(y_prob, y_true, y_uncert)
    compute_secondary(y_pred, y_true)

if __name__ == "__main__":
    main()
