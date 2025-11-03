from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader

from src.data.mvtec import MVTECImages, list_mvtec_images
from src.models.backbone_feats import ResNet18FeatureExtractor
from src.utils.seed import set_all_seeds

def build_feature_bank(data_root: str, category: str, image_size: int = 256,
                       batch_size: int = 32, device: str = "cpu", seed: int = 42,
                       save_path: str | None = None):
    """
    Extract 512-d features from train/good images and save a bank (.npz).
    """
    set_all_seeds(seed)
    train_paths, train_labels, _, _ = list_mvtec_images(data_root, category)
    ds = MVTECImages(train_paths, train_labels, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNet18FeatureExtractor(device=device)

    feats_list = []
    with torch.no_grad():
        for xb, yb, pb in tqdm(dl, desc=f"Extract train feats: {category}"):
            feats = model(xb)  # (B, 512)
            feats_list.append(feats.cpu().numpy())
    feats = np.concatenate(feats_list, axis=0)  # (N, 512)

    # Save
    if save_path is None:
        out_dir = Path("artifacts") / category
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "knn_bank.npz")
    np.savez_compressed(save_path, feats=feats, paths=np.array([str(p) for p in train_paths]))
    return feats, save_path

def score_test_split(data_root: str, category: str, bank_path: str,
                     image_size: int = 256, batch_size: int = 32,
                     device: str = "cpu", metric: str = "cosine"):
    """
    Load bank and compute anomaly scores for test images via kNN distance.
    Returns:
      - scores (np.array), labels (np.array), test_paths (list[str]), AUROC (float)
    """
    _, _, test_paths, test_labels = list_mvtec_images(data_root, category)
    ds = MVTECImages(test_paths, test_labels, image_size=image_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    bank = np.load(bank_path)
    bank_feats = bank["feats"]  # (N_train, 512)

    # Build NearestNeighbors index
    nn = NearestNeighbors(n_neighbors=1, metric=metric)
    nn.fit(bank_feats)

    model = ResNet18FeatureExtractor(device=device)

    scores = []
    labels = []
    paths = []
    with torch.no_grad():
        for xb, yb, pb in tqdm(dl, desc=f"Score test: {category}"):
            feats = model(xb).cpu().numpy()  # (B, 512)
            # distance to nearest neighbor in bank
            dists, _ = nn.kneighbors(feats, n_neighbors=1, return_distance=True)
            # anomaly score = distance (bigger = more anomalous)
            scores.extend(dists[:, 0].tolist())
            labels.extend(yb.numpy().tolist())
            paths.extend(list(pb))

    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)
    auroc = roc_auc_score(labels, scores)

    return scores, labels, paths, auroc
