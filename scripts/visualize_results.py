import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

def main():
    parser = argparse.ArgumentParser(description="Visualise top anomalous test images (image-level kNN).")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category, e.g., bottle")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Where results_eval.csv was saved")
    parser.add_argument("--topk", type=int, default=6, help="How many top anomalous images to show")
    args = parser.parse_args()

    csv_path = os.path.join(args.artifacts_dir, args.category, "results_eval.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}. Lance d'abord eval_knn pour créer le CSV.")

    df = pd.read_csv(csv_path)
    # df doit avoir colonnes: path, label, score
    if not {"path", "label", "score"}.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans {csv_path}. Colonnes trouvées: {df.columns.tolist()}")

    # normalisation des scores pour l'affichage
    scores = df["score"].values.astype(float).reshape(-1, 1)
    scl = MinMaxScaler()
    scores_norm = scl.fit_transform(scores).reshape(-1)

    df["score_norm"] = scores_norm
    # ordonner du plus anormal au moins anormal
    df_sorted = df.sort_values("score_norm", ascending=False).head(args.topk)

    n = len(df_sorted)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(4*ncols, 4*nrows))

    for i, row in enumerate(df_sorted.itertuples(), 1):
        p = row.path  # chemin tel qu'enregistré dans le CSV (souvent absolu)
        if not os.path.exists(p):
            # Petit fallback si certains chemins ont des séparateurs différents
            p_alt = os.path.normpath(p)
            if os.path.exists(p_alt):
                p = p_alt
            else:
                print(f"[WARN] Chemin introuvable, je skip: {p}")
                continue

        img = Image.open(p).convert("RGB")
        img_np = np.array(img).astype(float) / 255.0

        # Teinte rouge uniforme (image-level) proportionnelle au score (indicatif)
        alpha = min(max(row.score_norm, 0.0), 1.0)   # 0..1
        red_overlay = np.zeros_like(img_np)
        red_overlay[..., 0] = 1.0  # canal rouge

        blended = (1 - 0.35*alpha) * img_np + (0.35*alpha) * red_overlay

        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(blended)
        ax.set_title(f"{os.path.basename(p)}\nscore={row.score:.3f} / norm={row.score_norm:.2f}", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Top {n} images les plus anormales — {args.category} (kNN image-level)", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
