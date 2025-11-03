# run_all_categories.py
"""
Pipeline complet sur TOUTES les catégories MVTec AD :
- kNN : fit + eval
- AE  : train si besoin + eval
- Comparaisons + graphiques finaux

Exécuter :
    py run_all_categories.py
"""

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper"
]

def run(cmd: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("\n" + "="*80)
    print(f"➡️  Lancement : {cmd}")
    print("="*80)
    subprocess.run(cmd, shell=True, check=False, cwd=str(PROJECT_ROOT), env=env)

def has_data(cat: str) -> bool:
    return (DATA_ROOT / cat).exists()

def ensure_ae_trained(cat: str, device: str = "cpu", epochs: int = 10):
    """Entraîne l'AE si artifacts/<cat>/ae.pth est absent."""
    ckpt = ARTIFACTS / cat / "ae.pth"
    if ckpt.exists():
        return
    # Si ton repo a scripts/train_ae.py, on l'utilise (recommandé)
    # Sinon, adapte à ton script d'entraînement existant.
    run(f'py -m scripts.train_ae --data_root "{DATA_ROOT}" --category {cat} --epochs {epochs} --device {device}')
    # À la fin, scripts/train_ae doit avoir créé artifacts/<cat>/ae.pth

def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Le dossier data/ est introuvable : {DATA_ROOT}")

    ARTIFACTS.mkdir(exist_ok=True, parents=True)

    for cat in CATEGORIES:
        print(f"\n{'#'*80}\n📂 Catégorie : {cat}\n{'#'*80}")
        if not has_data(cat):
            print(f"⚠️  {cat} absente de data/ → on saute.")
            continue

        # kNN : fit + eval
        run(f'py -m scripts.fit_knn --data_root "{DATA_ROOT}" --category {cat}')
        run(f'py -m scripts.eval_knn --data_root "{DATA_ROOT}" --category {cat}')

        # AE : train si besoin + eval
        ensure_ae_trained(cat, device="cpu", epochs=10)
        run(f'py -m scripts.eval_ae --data_root "{DATA_ROOT}" --category {cat}')

    # Comparaison globale + fusion + graphique final
    print("\n📊 Comparaison globale (kNN vs AE)…")
    run('py -m scripts.compare_methods --save_roc')
    run('py -m scripts.merge_results')
    run('py -m scripts.plot_final')

    print("\n✅ Terminé. Résultats dans: artifacts/")

if __name__ == "__main__":
    main()
