# 🧠 Projet Final — Détection d’Anomalies sur le Dataset MVTec AD

## 🎯 Objectif du projet
L’objectif de ce projet est de **détecter automatiquement les anomalies visuelles** dans des images industrielles issues du **dataset MVTec AD**.  
Chaque catégorie (bottle, cable, hazelnut, etc.) contient des images **normales** et **défectueuses**.  
Le but est d’entraîner et de comparer plusieurs méthodes afin d’identifier les anomalies sans supervision directe.

👉 Le travail a été réalisé sur **toutes les catégories** du dataset MVTec AD, conformément aux consignes du projet.

---

## 💻 Prérequis

Avant d’exécuter le projet, il faut installer les outils suivants :

- Python 3.10 ou supérieur  
- PyTorch  
- scikit-learn  
- matplotlib  
- numpy, pandas  

Et télécharger jeu de donnée via ce lien :
 https://www.mvtec.com/company/research/datasets/mvtec-ad

 
### Installation rapide :
pip install -r requirements.txt
📁 Structure du projet
text
Copier le code
machine_learning/
├── data/                      # Dataset MVTec AD (toutes les catégories)
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── hazelnut/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│
├── src/                       # Code source du projet
│   ├── data/                  # Chargement et gestion des images
│   ├── models/                # Modèles (Autoencoder)
│   ├── methods/               # Méthodes kNN et reconstruction AE
│   └── utils/                 # Fonctions utilitaires
│
├── scripts/                   # Scripts exécutables
│   ├── visualize_examples.py
│   ├── fit_knn.py / eval_knn.py
│   ├── eval_ae.py / repeat_ae.py
│   ├── compare_methods.py / merge_results.py / plot_final.py
│   ├── visualize_heatmaps.py
│   └── eval_final.py
│
├── artifacts/                 # Résultats générés automatiquement
│   ├── bottle/
│   ├── cable/
│   ├── hazelnut/
│   ├── ...
│   ├── summary_compare.csv
│   ├── final_auroc_barplot.png
│   └── final_summary.csv
│
├── requirements.txt
└── README.md
🧩 Étape A — Visualisation du dataset
Afficher quelques exemples d’images normales et défectueuses :

py scripts/visualize_examples.py --data_root "data" --category bottle
➡️ Cette étape permet de comprendre la structure des données et d’observer les différences visuelles entre les images normales et les anomalies.

⚙️ Étape B — Méthode 1 : kNN sur features pré-entraînées
Entraînement :

py scripts/fit_knn.py --data_root "data" --category bottle
Évaluation :


py scripts/eval_knn.py --data_root "data" --category bottle
➡️ Répéter pour toutes les catégories :



py scripts/fit_knn.py --data_root "data" --category cable
py scripts/fit_knn.py --data_root "data" --category hazelnut
...
📄 Résultats :

Fichier : artifacts/<cat>/results_eval.csv

Contient les labels, scores et métriques de test.

🧠 Étape C — Méthode 2 : Autoencoder (AE)
Entraînement et évaluation :


py scripts/eval_ae.py --data_root "data" --category bottle
➡️ Répéter pour toutes les catégories :



py scripts/eval_ae.py --data_root "data" --category cable
py scripts/eval_ae.py --data_root "data" --category hazelnut
...
Vérification de la stabilité :


py scripts/repeat_ae.py --data_root "data" --categories bottle cable hazelnut metal_nut pill screw toothbrush transistor wood zipper --seeds 0 1 2 3 4 --epochs 10
📄 Résultats :

summary_ae_multiseed.csv → Moyenne ± écart-type

ae_heatmaps/ → visualisation des reconstructions

📊 Étape D — Comparaison des méthodes (kNN vs AE)
Comparer les performances sur toutes les catégories :



py scripts.compare_methods --save_roc
py scripts.merge_results
py scripts.plot_final
📈 Sorties :

summary_compare.csv

final_auroc_barplot.png (comparaison graphique des AUROC)

final_summary.csv

🔥 Étape E — Visualisation qualitative
Afficher les zones d’anomalies détectées (heatmaps) :



py scripts.visualize_heatmaps --category bottle
➡️ Répéter pour d’autres catégories :


py scripts.visualize_heatmaps --category cable
py scripts.visualize_heatmaps --category hazelnut
...
Ces cartes de chaleur montrent où le modèle détecte les anomalies dans les images.

✅ Étape F — Évaluation finale
Évaluer les performances finales du modèle choisi :



# Autoencoder
py scripts.eval_final --category bottle --method ae

# kNN
py scripts.eval_final --category bottle --method knn
➡️ Répéter pour toutes les catégories :



py scripts.eval_final --category cable --method ae
py scripts.eval_final --category hazelnut --method ae
...
📊 Fichiers générés :

roc_ae.png, pr_ae.png, confmat_ae.png

top_ae.png, faux_positifs.png, faux_négatifs.png

final_report_ae.csv → rapport complet avec métriques et figures

📈 Interprétation des résultats
Catégorie	Méthode	AUROC	AP	Accuracy	F1	Commentaire
Bottle	AE	~0.97	~0.95	0.93	0.91	Très bonne reconstruction
Cable	kNN	~0.85	~0.80	0.86	0.82	Méthode simple mais stable
Hazelnut	AE	~0.98	~0.96	0.94	0.93	Haute précision
...	...	...	...	...	...	...

🧩 Ces valeurs peuvent légèrement varier selon les seeds ou les paramètres d’entraînement.

🧠 Points importants à retenir
Les modèles sont entraînés uniquement sur les images normales (train/good).

Les images défectueuses ne sont utilisées qu’en phase de test.

Les métriques principales :

AUROC : Aire sous la courbe ROC (mesure de la qualité du classement)

AP : Average Precision (équilibre précision/rappel)

Accuracy et F1-score : qualité de la classification finale

Tous les résultats et figures sont enregistrés automatiquement dans artifacts/.

💡 Conclusion
Ce projet démontre :

la préparation et le traitement d’un dataset industriel d’anomalies visuelles (MVTec AD),

la comparaison entre deux approches :

une méthode simple et robuste (kNN),

une méthode neurale non supervisée (Autoencoder),

l’importance de l’évaluation multi-catégorie et multi-initialisation,

la capacité à visualiser et interpréter les anomalies détectées.

Les scripts fournis garantissent une reproductibilité complète et une analyse rigoureuse des performances sur toutes les catégories du dataset.
