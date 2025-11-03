import argparse, os, glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Show AE heatmaps saved during eval")
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    hm_dir = Path(args.artifacts_dir) / args.category / "ae_heatmaps"
    if not hm_dir.exists():
        raise FileNotFoundError(f"Pas de heatmaps trouvées dans: {hm_dir}. Lance eval_ae d'abord.")

    hms = sorted(glob.glob(str(hm_dir / "hm_*.png")))
    hms = hms[:args.n]
    if not hms:
        print("Aucune heatmap à afficher.")
        return

    cols = 3 if len(hms) >= 3 else len(hms)
    rows = int(np.ceil(len(hms)/cols))
    plt.figure(figsize=(4*cols, 4*rows))

    for i, hm_path in enumerate(hms, 1):
        # retrouver l'image d'origine (nom identique) pour overlay
        img_name = os.path.basename(hm_path).replace("hm_", "")
        # on suppose qu'elle est dans data/<cat>/test/.../<img_name>
        # si besoin, juste afficher la heatmap seule
        hm = np.array(Image.open(hm_path).convert("L")) / 255.0
        ax = plt.subplot(rows, cols, i)
        ax.imshow(hm, cmap="hot")
        ax.set_title(img_name)
        ax.axis("off")

    plt.suptitle(f"AE heatmaps — {args.category}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
