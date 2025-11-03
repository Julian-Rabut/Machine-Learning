# scripts/repeat_ae.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import shutil

from src.methods.ae_recon import train_autoencoder, evaluate_autoencoder

def main():
    p = argparse.ArgumentParser(description="Multi-runs AE (plusieurs seeds) + agrégation AUROC")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--categories", nargs="+", required=True, help="ex: bottle cable hazelnut")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = p.parse_args()

    rows = []
    for cat in args.categories:
        print(f"\n=== Catégorie: {cat} ===")
        cat_dir = Path(args.artifacts_dir) / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            # chemins par seed
            ckpt_seed = cat_dir / f"ae_seed{seed}.pth"
            csv_seed  = cat_dir / f"results_eval_ae_seed{seed}.csv"

            # 1) train
            print(f"[Train] {cat} | seed={seed}")
            ckpt_tmp = train_autoencoder(
                data_root=args.data_root,
                category=cat,
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                save_dir=args.artifacts_dir,
                seed=seed
            )
            # renommer le checkpoint spécifique à ce seed
            Path(ckpt_tmp).rename(ckpt_seed)

            # 2) eval
            print(f"[Eval ] {cat} | seed={seed}")
            au, csv_out, _ = evaluate_autoencoder(
                data_root=args.data_root,
                category=cat,
                ckpt_path=str(ckpt_seed),
                image_size=args.image_size,
                batch_size=max(8, args.batch_size//2),
                device=args.device,
                save_dir=args.artifacts_dir
            )
            # renommer le csv spécifique à ce seed (si besoin)
            Path(csv_out).rename(csv_seed)

            rows.append({"category": cat, "seed": seed, "method": "AE", "auroc": au})

        # agrégation par catégorie
        df_cat = pd.DataFrame([r for r in rows if r["category"] == cat])
        mean = df_cat["auroc"].mean()
        std  = df_cat["auroc"].std(ddof=1) if len(df_cat) > 1 else 0.0
        print(f"[Agg ] {cat} | AE AUROC = {mean:.4f} ± {std:.4f}  (n={len(df_cat)})")

    # sauvegarde récap global
    out_csv = Path(args.artifacts_dir) / "summary_ae_multiseed.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] Récap multi-seed écrit dans: {out_csv}")

if __name__ == "__main__":
    main()
