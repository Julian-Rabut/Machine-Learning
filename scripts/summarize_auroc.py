# scripts/summarize_auroc.py
import argparse, os, glob, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

def auroc_from_csv(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    # robustesse: colonnes attendues path,label,score
    y = df["label"].astype(int).values
    s = df["score"].astype(float).values
    return roc_auc_score(y, s)

def main():
    parser = argparse.ArgumentParser(description="Résumé AUROC par catégorie à partir de artifacts/*/results_eval.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Dossier racine des résultats")
    parser.add_argument("--out_csv", type=str, default="artifacts/summary_auroc.csv")
    parser.add_argument("--out_png", type=str, default="artifacts/summary_auroc.png")
    args = parser.parse_args()

    rows = []
    for cat_dir in sorted(glob.glob(os.path.join(args.artifacts_dir, "*"))):
        if not os.path.isdir(cat_dir):
            continue
        csv_path = os.path.join(cat_dir, "results_eval.csv")
        if not os.path.exists(csv_path):
            continue
        cat = os.path.basename(cat_dir)
        try:
            auroc = auroc_from_csv(csv_path)
            rows.append({"category": cat, "auroc": auroc, "csv": csv_path})
        except Exception as e:
            print(f"[WARN] AUROC échoué pour {csv_path}: {e}")

    if not rows:
        print("Aucun results_eval.csv trouvé. Lance d'abord eval_knn sur au moins une catégorie.")
        return

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("\n=== Résumé AUROC (kNN features) ===")
    print(df.to_string(index=False))
    print(f"\n[OK] Sauvegardé: {args.out_csv}")

    # Bar plot
    plt.figure(figsize=(8, 4))
    plt.bar(df["category"], df["auroc"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("AUROC (image-level)")
    plt.title("kNN features — AUROC par catégorie")
    for i, v in enumerate(df["auroc"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"[OK] Graphique sauvegardé: {args.out_png}")

if __name__ == "__main__":
    main()
