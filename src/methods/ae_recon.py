import os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
from src.utils.seed import set_all_seeds
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- Dataset (AE) : on garde les images en [0,1], pas de normalisation ImageNet ----------
class MVTECImagesAE(Dataset):
    def __init__(self, paths: List[Path], image_size: int = 256):
        self.paths = list(paths)
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # -> [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)      # (3,H,W) in [0,1]
        return x, str(p)

# ---------- Petit Autoencoder convolutionnel ----------
class ConvAE(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        # 256 -> 128 -> 64 -> 32 -> 16 (downsample x16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1), nn.ReLU(inplace=True),    # 256->128
            nn.Conv2d(base, base*2, 4, 2, 1), nn.ReLU(inplace=True),         # 128->64
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.ReLU(inplace=True),       # 64->32
            nn.Conv2d(base*4, base*4, 4, 2, 1), nn.ReLU(inplace=True),       # 32->16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*4, 4, 2, 1), nn.ReLU(inplace=True),  # 16->32
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.ReLU(inplace=True),  # 32->64
            nn.ConvTranspose2d(base*2, base,   4, 2, 1), nn.ReLU(inplace=True),  # 64->128
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1), nn.Sigmoid(),        # 128->256
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

# ---------- Utils : lister les chemins MVTec ----------
def list_mvtec_split_paths(data_root: str, category: str):
    """Retourne (train_good_paths, test_paths, test_labels) avec labels 0/1"""
    root = Path(data_root) / category
    train_good = sorted((root / "train" / "good").rglob("*.png")) + \
                 sorted((root / "train" / "good").rglob("*.jpg"))

    test_paths, test_labels = [], []
    test_root = root / "test"
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        imgs = sorted(list(sub.rglob("*.png")) + list(sub.rglob("*.jpg")))
        if sub.name == "good":
            test_paths += imgs;  test_labels += [0]*len(imgs)
        else:
            test_paths += imgs;  test_labels += [1]*len(imgs)
    return train_good, test_paths, np.array(test_labels, dtype=int)

# ---------- Entraînement ----------
def train_autoencoder(data_root: str, category: str, image_size=256, epochs=10,
                      batch_size=16, lr=1e-3, device="cpu", save_dir="artifacts", seed: int = 42):
    set_all_seeds(seed)

    outdir = Path(save_dir) / category
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = outdir / "ae.pth"
    ...

    train_good, _, _ = list_mvtec_split_paths(data_root, category)
    ds = MVTECImagesAE(train_good, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        for xb, _ in dl:
            xb = xb.to(device)
            xhat = model(xb)
            loss = crit(xhat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        print(f"[AE][{category}] epoch {ep:02d}/{epochs}  loss={running/len(ds):.5f}")

    outdir = Path(save_dir) / category
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / "ae.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"[OK] AE sauvegardé: {ckpt}")
    return str(ckpt)

# ---------- Évaluation : scores image-level + heatmaps ----------
@torch.no_grad()
def evaluate_autoencoder(data_root: str, category: str, ckpt_path: str,
                         image_size=256, batch_size=8, device="cpu",
                         save_dir="artifacts"):
    _, test_paths, test_labels = list_mvtec_split_paths(data_root, category)
    ds = MVTECImagesAE(test_paths, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConvAE().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    import pandas as pd
    from sklearn.metrics import roc_auc_score
    outdir = Path(save_dir) / category
    outdir.mkdir(parents=True, exist_ok=True)
    csv_out = outdir / "results_eval_ae.csv"

    scores, paths = [], []
    # pour heatmaps (on en sauvegarde quelques-unes)
    hm_dir = outdir / "ae_heatmaps"
    hm_dir.mkdir(exist_ok=True)

    for xb, pbatch in dl:
        xb = xb.to(device)
        xhat = model(xb)
        # erreur pixel-wise (par image)
        err = (xb - xhat).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
        # score image-level = moyenne des erreurs pixels
        img_scores = err.mean(dim=(1,2,3)).cpu().numpy().tolist()
        scores.extend(img_scores)
        paths.extend(list(pbatch))

        # sauver quelques heatmaps
        for i in range(min(len(pbatch), 2)):  # 2 par batch pour ne pas tout remplir
            e = err[i,0].cpu().numpy()
            e = (e - e.min()) / (e.max() - e.min() + 1e-8)
            import matplotlib.pyplot as plt
            plt.imsave(hm_dir / f"hm_{os.path.basename(pbatch[i])}", e, cmap="hot")

    scores = np.array(scores, dtype=float)
    y = test_labels.astype(int)
    auroc = roc_auc_score(y, scores)

    pd.DataFrame({"path": paths, "label": y, "score": scores}).to_csv(csv_out, index=False)
    print(f"[OK] AE image-level AUROC (test) = {auroc:.4f}")
    print(f"[OK] CSV: {csv_out} | heatmaps dans: {hm_dir}")
    return auroc, str(csv_out), str(hm_dir)
