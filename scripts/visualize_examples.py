import argparse
from src.data.mvtec import list_mvtec_images
from src.utils.viz import show_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick visualization of MVTec images")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--n", type=int, default=8, help="number of images per split to preview")
    args = parser.parse_args()

    train_paths, _, test_paths, _ = list_mvtec_images(args.data_root, args.category)
    show_images(train_paths[:args.n], title=f"{args.category} — train/good (first {args.n})")
    show_images(test_paths[:args.n],  title=f"{args.category} — test (first {args.n})")
