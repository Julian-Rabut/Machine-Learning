# scripts/eval_knn.py
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_recall_fscore_support
)

from src.methods.knn_distance import score_test_split


def best_threshold_youden(y_val, s_val):
    """
    Choix du seuil par le critère de Youden: J = TPR - FPR.
    Retourne: threshold*, tpr*, fpr*
    """
    fpr, tpr, thr = roc_curve(y_val, s_val)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])


def best_threshold_max_f1(y_val, s_val):
    """
    Balayage des seuils issus de la courbe ROC pour maximiser le F1.
    Retourne: threshold*, f1*
    """
    fpr, tpr, thr = roc_curve(y_val, s_val)
    best_f1, best_thr = -1.0, float(thr[0])
    for t in thr:
        yhat = (s_val >= t).astype(int)
        f1 = f1_score(y_val, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate kNN anomaly score on test split")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec AD root")
    parser.add_argument("--category", type=str, required=True, help="MVTec category, e.g., bottle")
    parser.add_argument("--bank", type=str, default=None,
                        help="Path to saved bank .npz (default: artifacts/{cat}/knn_bank.npz)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    # Optionnel: choisir la règle de seuil
    parser.add_argument("--threshold_rule", type=str, default="youden", choices=["youden", "max_f1"],
                        help="Règle pour choisir le seuil sur la partie validation interne")
    # Optionnel: ratio validation dans le split (sur le test pour simuler une val)
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Part du test utilisée pour choisir le seuil")
    args = parser.parse_args()

    if args.bank is None:
        args.bank = str(Path("artifacts") / args.category / "knn_bank.npz")

    # 1) Scores kNN + AUROC (comme avant)
    scores, labels, paths, auroc = score_test_split(
        data_root=args.data_root,
        category=args.category,
        bank_path=args.bank,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        metric=args.metric,
    )

    out_dir = Path("artifacts") / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des scores par image (comme avant)
    out_csv = out_dir / "results_eval.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score"])
        for p, y, s in zip(paths, labels, scores):
            w.writerow([p, int(y), f"{float(s):.6f}"])

    print(f"[OK] AUROC (image-level) on test = {auroc:.4f}")
    print(f"[OK] Saved per-image scores to: {out_csv}")

    # 2) Sélection d’un seuil sur une pseudo-validation (split du test)
    #    - BUT: montrer qu’on a CHERCHÉ un seuil, pas fixé par FP/FN cibles
    #    - Méthode simple et reproductible
    n = len(labels)
    idx = np.arange(n)
    rng = np.random.default_rng(42)  # seed fixe pour reproductibilité
    rng.shuffle(idx)

    n_val = max(1, int(args.val_ratio * n))
    val_idx = idx[:n_val]
    tst_idx = idx[n_val:]

    y_val, s_val = labels[val_idx], scores[val_idx]
    y_tst, s_tst = labels[tst_idx], scores[tst_idx]

    if args.threshold_rule == "youden":
        thr_star, tpr_v, fpr_v = best_threshold_youden(y_val, s_val)
    else:
        thr_star, best_f1_val = best_threshold_max_f1(y_val, s_val)

    # 3) Évaluation sur la partie test (tenue à l’écart du choix du seuil)
    yhat_tst = (s_tst >= thr_star).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_tst, yhat_tst, average="binary", zero_division=0
    )

    print("[INFO] Threshold selection")
    print(f"      - Rule: {args.threshold_rule}")
    print(f"      - Threshold* = {thr_star:.6f}")
    if args.threshold_rule == "youden":
        print(f"      - On validation: TPR={tpr_v:.3f} | FPR={fpr_v:.3f}")
    else:
        print(f"      - On validation: best F1 = {best_f1_val:.3f}")
    print(f"[INFO] On held-out test: Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")

    # 4) Sauvegarde d’un petit rapport JSON
    report_path = out_dir / "threshold_report.json"
    payload = {
        "category": args.category,
        "metric": args.metric,
        "auroc_image": float(auroc),
        "threshold_rule": args.threshold_rule,
        "val_ratio": float(args.val_ratio),
        "threshold": float(thr_star),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "n_val_used": int(len(val_idx)),
        "n_test_used": int(len(tst_idx)),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] Rapport sauvegardé: {report_path}")
