# scripts/plot_final.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("artifacts/final_summary.csv")
    df_ae = df[df["method"].str.contains("AE", case=False)].copy()
    df_knn = df[df["method"].str.contains("knn", case=False)].copy()

    # Fusion pour aligner kNN / AE par catégorie
    merged = pd.merge(df_knn[["category", "mean"]],
                      df_ae[["category", "mean", "std"]],
                      on="category", suffixes=("_knn", "_ae"))

    # Graphique
    x = range(len(merged))
    plt.figure(figsize=(10,5))
    plt.bar(x, merged["mean_knn"], width=0.4, label="kNN", color="steelblue")
    plt.bar([i+0.4 for i in x], merged["mean_ae"], width=0.4, 
            yerr=merged["std"], capsize=4, label="AE (multi-seed)", color="orange")

    plt.xticks([i+0.2 for i in x], merged["category"], rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.title("Comparaison kNN vs AutoEncoder (moyenne ± écart-type)")
    plt.legend()
    plt.tight_layout()

    out_path = Path("artifacts/final_auroc_barplot.png")
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Graphique sauvegardé: {out_path}")

if __name__ == "__main__":
    main()
