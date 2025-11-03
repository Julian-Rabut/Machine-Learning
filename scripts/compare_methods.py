# scripts/compare_methods.py
import argparse, os, glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_scores(csv_path):
    """Lit un CSV (colonnes: path,label,score) -> y, s, avec fallback encodage."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")
    if not {"label", "score"}.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans {csv_path}.")
    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    return y, s


def compute_auroc(y, s):
    return roc_auc_score(y, s)

def plot_roc(ax, y, s, label):
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, label=f"{label} (AUROC={roc_auc_score(y,s):.3f})")

def main():
    p = argparse.ArgumentParser(description="Compare kNN vs AE (AUROC + ROC) pour une ou plusieurs catégories")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Liste de catégories; si vide: détecte automatiquement dans artifacts/")
    p.add_argument("--out_csv", type=str, default="artifacts/summary_compare.csv")
    p.add_argument("--save_roc", action="store_true", help="Sauvegarde aussi des figures ROC par catégorie")
    args = p.parse_args()

    # détecter les catégories si non spécifiées
    if not args.categories:
        cats = [os.path.basename(p) for p in glob.glob(os.path.join(args.artifacts_dir,"*"))
                if os.path.isdir(p)]
    else:
        cats = args.categories

    rows = []
    for cat in sorted(cats):
        cat_dir = Path(args.artifacts_dir) / cat
        knn_csv = cat_dir / "results_eval.csv"
        ae_csv  = cat_dir / "results_eval_ae.csv"
        if not knn_csv.exists() and not ae_csv.exists():
            continue

        fig, ax = None, None
        if args.save_roc:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot([0,1],[0,1],"k--",alpha=0.4)
            ax.set_title(f"ROC — {cat}")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

        if knn_csv.exists():
            y_k, s_k = load_scores(str(knn_csv))
            au_k = compute_auroc(y_k, s_k)
            rows.append({"category": cat, "method": "kNN", "auroc": au_k})
            if args.save_roc:
                plot_roc(ax, y_k, s_k, "kNN")

        if ae_csv.exists():
            y_a, s_a = load_scores(str(ae_csv))
            au_a = compute_auroc(y_a, s_a)
            rows.append({"category": cat, "method": "AE", "auroc": au_a})
            if args.save_roc:
                plot_roc(ax, y_a, s_a, "AE")

        if args.save_roc:
            ax.legend()
            fig.tight_layout()
            out_png = cat_dir / "roc_compare.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[OK] ROC sauvegardée: {out_png}")

    if not rows:
        print("Aucun résultat trouvé. Lance d'abord les évaluations (kNN/AE).")
        return

    df = pd.DataFrame(rows).sort_values(["category","method"])
    Path(args.artifacts_dir).mkdir(exist_ok=True, parents=True)
    df.to_csv(args.out_csv, index=False)
    print("\n=== Comparaison AUROC (kNN vs AE) ===")
    print(df.to_string(index=False))
    print(f"\n[OK] CSV récap: {args.out_csv}")

    # Petit récap global par méthode (moyenne ± écart-type)
    summary = df.groupby("method")["auroc"].agg(["mean","std"]).reset_index()
    print("\nMoyenne ± écart-type (toutes catégories):")
    for r in summary.itertuples():
        print(f"- {r.method}: {r.mean:.3f} ± {r.std:.3f}")

if __name__ == "__main__":
    main()
