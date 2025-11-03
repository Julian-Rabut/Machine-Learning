# scripts/visualize_heatmaps.py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

def visualize_heatmaps(category="bottle", artifacts_dir="artifacts", n=6):
    """
    Affiche quelques heatmaps sauvegardées par eval_ae.py
    (artifacts/<category>/ae_heatmaps/hm_*.png).
    """
    hm_dir = Path(artifacts_dir) / category / "ae_heatmaps"
    if not hm_dir.exists():
        raise FileNotFoundError(
            f"Dossier {hm_dir} introuvable.\n"
            f"→ Lance d'abord : py -m scripts.eval_ae --data_root data --category {category}"
        )

    hms = sorted(glob.glob(str(hm_dir / "hm_*.png")))
    if not hms:
        raise FileNotFoundError(
            f"Aucune heatmap trouvée dans {hm_dir}.\n"
            f"→ Relance l'évaluation AE pour en générer."
        )

    hms = hms[:n]
    cols = min(3, len(hms))
    rows = int(np.ceil(len(hms) / cols))

    plt.figure(figsize=(4*cols, 4*rows))
    for i, hm_path in enumerate(hms, 1):
        hm = np.array(Image.open(hm_path).convert("L")) / 255.0
        ax = plt.subplot(rows, cols, i)
        ax.imshow(hm, cmap="hot")
        ax.set_title(os.path.basename(hm_path).replace("hm_",""))
        ax.axis("off")

    plt.suptitle(f"AE heatmaps — {category}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()
    visualize_heatmaps(args.category, args.artifacts_dir, args.n)
