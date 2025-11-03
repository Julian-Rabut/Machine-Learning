import pandas as pd
import numpy as np
from pathlib import Path

def main():
    ae_path = Path("artifacts/summary_ae_multiseed.csv")
    cmp_path = Path("artifacts/summary_compare.csv")
    out_path = Path("artifacts/final_summary.csv")

    if not ae_path.exists():
        raise FileNotFoundError(f"Manque {ae_path}")
    if not cmp_path.exists():
        raise FileNotFoundError(f"Manque {cmp_path}")

    df_ae = pd.read_csv(ae_path)
    df_cmp = pd.read_csv(cmp_path)

    # ----- AE: agrégation multi-seed -----
    if not set(["category","auroc"]).issubset(df_ae.columns):
        raise ValueError(f"'summary_ae_multiseed.csv' doit contenir ['category','auroc']. Colonnes: {df_ae.columns.tolist()}")

    df_agg = (
        df_ae.groupby("category")["auroc"]
             .agg(mean="mean", std=lambda s: float(np.std(s, ddof=1)) if len(s)>1 else 0.0)
             .reset_index()
    )
    df_agg["method"] = "AE (multi-seed)"
    df_agg["mean_std"] = df_agg["mean"].round(3).astype(str) + " ± " + df_agg["std"].round(3).astype(str)

    # ----- kNN: extraire colonne AUROC, nom variable -----
    if "method" not in df_cmp.columns or "category" not in df_cmp.columns:
        raise ValueError(f"'summary_compare.csv' doit contenir ['category','method']. Colonnes: {df_cmp.columns.tolist()}")

    auroc_cols = [c for c in df_cmp.columns if c.lower().startswith("auroc")]
    if not auroc_cols:
        raise ValueError(f"Aucune colonne AUROC trouvée dans {cmp_path}. Colonnes: {df_cmp.columns.tolist()}")
    cmp_auroc_col = auroc_cols[0]  # ex. 'auroc'

    df_knn = df_cmp[df_cmp["method"].str.contains("knn", case=False)].copy()
    if df_knn.empty:
        # fallback: tout ce qui n'est pas AE
        df_knn = df_cmp[~df_cmp["method"].str.contains("ae", case=False)].copy()

    df_knn = df_knn[["category","method",cmp_auroc_col]].rename(columns={cmp_auroc_col:"mean"})
    df_knn["mean_std"] = df_knn["mean"].round(3).astype(str)

    # ----- Fusion finale -----
    df_final = pd.concat([
        df_knn[["category","method","mean","mean_std"]],
        df_agg[["category","method","mean","std","mean_std"]]
    ], ignore_index=True).sort_values(["category","method"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    print(df_final.to_string(index=False))
    print(f"\n[OK] Sauvegardé: {out_path}")

if __name__ == "__main__":
    main()
