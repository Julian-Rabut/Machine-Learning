import argparse
from src.methods.ae_recon import train_autoencoder

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train AE on MVTec train/good")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--category", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    args = p.parse_args()

    train_autoencoder(
        data_root=args.data_root,
        category=args.category,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
