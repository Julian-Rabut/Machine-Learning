import argparse
from pathlib import Path
from src.methods.knn_distance import build_feature_bank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build kNN feature bank from train/good images")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec AD root")
    parser.add_argument("--category", type=str, required=True, help="MVTec category, e.g., bottle")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Optional: custom output .npz path")
    args = parser.parse_args()

    feats, save_path = build_feature_bank(
        data_root=args.data_root,
        category=args.category,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        save_path=args.out,
    )
    print(f"[OK] Saved feature bank to: {save_path}")
