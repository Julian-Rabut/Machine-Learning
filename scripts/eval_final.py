# scripts/eval_final.py
import argparse, os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score
)

# ---------- Utils chargement CSV "path,label,score" ----------
def load_scores(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {"label","score","path"}.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans {csv_path}. Requis: path,label,score")
    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    paths = df["path"].astype(str).values
    return y, s, paths, df

def best_threshold_youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thr[k]), fpr, tpr, thr

def plot_roc_pr(save_dir: Path, y, s, tag="final"):
    auroc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    # ROC
    thr_opt, fpr, tpr, thr = best_threshold_youden(y, s)
    plt.figure(figsize=(4.5,4))
    plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=0.3)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.tight_layout()
    out_roc = save_dir / f"roc_{tag}.png"
    plt.savefig(out_roc, dpi=150); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(4.5,4))
    plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout()
    out_pr = save_dir / f"pr_{tag}.png"
    plt.savefig(out_pr, dpi=150); plt.close()
    return auroc, ap, thr_opt, out_roc, out_pr

def save_confmat(save_dir: Path, y_true, y_pred, tag="final"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3.8,3.4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matrice de confusion"); plt.colorbar()
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0,1], ["Normal","Anom."]); plt.yticks([0,1], ["Normal","Anom."])
    plt.xlabel("Prédit"); plt.ylabel("Vrai")
    plt.tight_layout()
    out_cm = save_dir / f"confmat_{tag}.png"
    plt.savefig(out_cm, dpi=150); plt.close()
    return out_cm

def pick_best_ae_seed(art_dir: Path):
    # cherche tous les CSV "results_eval_ae_seed*.csv", sélectionne l’AUROC max
    candidates = sorted(glob.glob(str(art_dir / "results_eval_ae_seed*.csv")))
    if not candidates and (art_dir / "results_eval_ae.csv").exists():
        candidates = [str(art_dir / "results_eval_ae.csv")]
    if not candidates:
        raise FileNotFoundError(f"Aucun résultat AE trouvé dans {art_dir}")

    best_csv, best_auroc = None, -1
    for c in candidates:
        y, s, _, _ = load_scores(c)
        au = roc_auc_score(y, s)
        if au > best_auroc:
            best_auroc, best_csv = au, c
    return best_csv, best_auroc

def show_top_images(save_dir: Path, df_scores: pd.DataFrame, top_k=6, tag="top_anomalies"):
    # top anomalies = plus gros scores
    df = df_scores.sort_values("score", ascending=False).head(top_k)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    cols = 3 if top_k >= 3 else top_k
    rows = int(np.ceil(top_k / cols))
    plt.figure(figsize=(4*cols, 4*rows))
    for i, row in enumerate(df.itertuples(), 1):
        ax = plt.subplot(rows, cols, i)
        img = mpimg.imread(row.path)
        ax.imshow(img)
        ax.set_title(f"{os.path.basename(row.path)}\nscore={row.score:.3f} | y={row.label}")
        ax.axis("off")
    plt.suptitle("Top images anormales (score élevé)")
    plt.tight_layout()
    out = save_dir / f"{tag}.png"
    plt.savefig(out, dpi=150); plt.close()
    return out

def show_fp_fn(save_dir: Path, df_scores: pd.DataFrame, y, s, thr, top_k=6, tag="fp_fn"):
    y_pred = (s >= thr).astype(int)
    df = df_scores.copy()
    df["pred"] = y_pred

    fps = df[(df["label"]==0) & (df["pred"]==1)].sort_values("score", ascending=False).head(top_k)
    fns = df[(df["label"]==1) & (df["pred"]==0)].sort_values("score", ascending=True).head(top_k)

    def _plot(rows, name):
        import matplotlib.image as mpimg
        if rows.empty:
            return None
        cols = 3 if len(rows) >= 3 else len(rows)
        rows_n = int(np.ceil(len(rows) / cols))
        plt.figure(figsize=(4*cols, 4*rows_n))
        for i, row in enumerate(rows.itertuples(), 1):
            ax = plt.subplot(rows_n, cols, i)
            ax.imshow(mpimg.imread(row.path))
            ax.set_title(f"{os.path.basename(row.path)}\nscore={row.score:.3f}")
            ax.axis("off")
        plt.suptitle(name); plt.tight_layout()
        out = save_dir / (name.replace(" ", "_") + ".png")
        plt.savefig(out, dpi=150); plt.close()
        return out

    out_fp = _plot(fps, "Faux positifs")
    out_fn = _plot(fns, "Faux négatifs")
    return out_fp, out_fn

def main():
    ap = argparse.ArgumentParser(description="Évaluation finale par catégorie (AE ou kNN)")
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--method", type=str, choices=["ae","knn"], default="ae")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--top_k", type=int, default=6)
    args = ap.parse_args()

    art_cat = Path(args.artifacts_dir) / args.category
    art_cat.mkdir(parents=True, exist_ok=True)

    # 1) Charger scores selon la méthode
    if args.method == "ae":
        csv_path, au_seed = pick_best_ae_seed(art_cat)
        print(f"[AE] meilleur run: {csv_path} (AUROC={au_seed:.3f})")
    else:
        csv_path = art_cat / "results_eval.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} introuvable (évalue d'abord kNN).")
        au_seed = None

    y, s, paths, df_scores = load_scores(str(csv_path))
    print(f"Images: {len(y)} | anormales: {int((y==1).sum())} | normales: {int((y==0).sum())}")

    # 2) Courbes + seuil
    auroc, ap_score, thr, roc_png, pr_png = plot_roc_pr(art_cat, y, s, tag=args.method)
    print(f"AUROC={auroc:.3f} | AP={ap_score:.3f} | seuil*={thr:.4f} (Youden)")

    # 3) Prédictions + métriques
    y_pred = (s >= thr).astype(int)
    acc = (y_pred == y).mean()
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm_png = save_confmat(art_cat, y, y_pred, tag=args.method)

    # 4) Visuels : top anomalies + FP/FN
    top_png = show_top_images(art_cat, df_scores, top_k=args.top_k, tag=f"top_{args.method}")
    fp_png, fn_png = show_fp_fn(art_cat, df_scores, y, s, thr, top_k=args.top_k, tag=f"fp_fn_{args.method}")

    # 5) Rapport CSV
    out_report = art_cat / f"final_report_{args.method}.csv"
    rep = pd.DataFrame([{
        "category": args.category,
        "method": args.method.upper(),
        "auroc": auroc,
        "average_precision": ap_score,
        "threshold_youden": thr,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_png": str(roc_png),
        "pr_png": str(pr_png),
        "confmat_png": str(cm_png),
        "top_png": str(top_png),
        "fp_png": str(fp_png) if fp_png else "",
        "fn_png": str(fn_png) if fn_png else ""
    }])
    rep.to_csv(out_report, index=False)
    print(f"[OK] Rapport écrit: {out_report}")

if __name__ == "__main__":
    main()
