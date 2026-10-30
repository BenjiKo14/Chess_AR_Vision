# Spec & Plan d'Implémentation — Slice 5 : Collecte Auto-Labellisée, Fine-Tuning GPU (`MobileNetV3`) & Inférence CPU (`ONNX`)

**Date** : 2026-10-30  
**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Prérequis** : Slices 1 & 2 (et optionnellement 3)  
**Objectif** : Permettre de reconnaître **n'importe quelle position d'échecs en milieu de partie (Cold Start / Full FEN Recognition)** avec une précision >99% sur l'échiquier et les pièces de l'utilisateur, en entraînant sur GPU un modèle `MobileNetV3-Small` puis en l'exécutant en **100% CPU via ONNX (< 50 ms pour les 64 cases)** sur PC portable ou Raspberry Pi.

---

## 1. Objectif & Contexte

L'Approche B (Slice 2) est imbattable pour suivre les coups à partir d'une position connue, mais elle ne peut pas deviner l'identité des pièces si on allume la caméra **en plein milieu d'une partie déjà commencée**.  
À l'inverse, un modèle pré-entraîné générique (comme `richardhbtz/chess-vision`) se trompe souvent sur un jeu de pièces de salon qu'il n'a jamais vu.

Cette **Slice 5** combine le meilleur des deux mondes grâce à un pipeline en 3 étapes ultra-rapide :
1. **Collecteur de Dataset Auto-Labellisé (`dataset_collector.py` + UI Web)** :
   - Profite du fait que `chess_vision.py` découpe déjà les 64 cases (`warped` 512×512) et que `chess_game.py` connaît le FEN courant (`python-chess`).
   - En un clic sur **"📸 Capturer les 64 cases dans le Dataset"** (ou automatiquement à chaque coup validé pendant une partie d'entraînement), le système découpe les 64 vignettes et les enregistre automatiquement dans les 13 dossiers correspondants (`dataset/empty/`, `dataset/wP/`, ..., `dataset/bK/`).
   - **Zéro labellisation manuelle** : en 10 à 15 captures (position initiale + quelques coups + changements légers d'éclairage/angle), on obtient 640 à 1000 vignettes parfaitement labellisées !
2. **Script de Fine-Tuning GPU & Export ONNX (`train_piece_classifier.py`)** :
   - S'exécute sur le PC équipé d'un GPU (PyTorch + Torchvision).
   - Fine-tune `mobilenet_v3_small` (pré-entraîné ImageNet) sur les 13 classes avec *Data Augmentation* (jitter de luminosité/contraste, légère translation/zoom pour simuler un léger décalage de perspective).
   - Applique un rééquilibrage des classes (`WeightedRandomSampler` ou `CrossEntropyLoss(weight=...)`) car dans une partie d'échecs il y a beaucoup plus de cases `empty` et de `pions` que de `Rois` ou de `Dames` !
   - Exporte le meilleur modèle au format universel **`models/chess_mobilenetv3.onnx`** (~5 Mo).
3. **Reconnaissance Instantanée en Milieu de Partie sur CPU (`onnx_classifier.py`)** :
   - Charge `models/chess_mobilenetv3.onnx` avec **`cv2.dnn.readNetFromONNX`** (inclus dans OpenCV de base, **aucune dépendance à PyTorch ni GPU requise sur le PC portable ou le Raspberry Pi** !).
   - Passe les 64 cases en un seul batch `(64, 3, 96, 96)` en **~35 ms sur CPU**.
   - Applique les règles de cohérence d'échecs (exactement 1 Roi blanc et 1 Roi noir, aucun pion sur les rangées 1 et 8) en utilisant les probabilités Softmax pour corriger automatiquement les éventuelles hésitations, et génère la chaîne **FEN complète** pour démarrer ou resynchroniser la partie à n'importe quel moment !

---

## 2. Les 13 Classes Standardisées

```python
CLASS_NAMES = [
    "empty",  # 0 : Case vide (blanche ou noire)
    "wP",     # 1 : Pion Blanc
    "wN",     # 2 : Cavalier Blanc
    "wB",     # 3 : Fou Blanc
    "wR",     # 4 : Tour Blanche
    "wQ",     # 5 : Dame Blanche
    "wK",     # 6 : Roi Blanc
    "bP",     # 7 : Pion Noir
    "bN",     # 8 : Cavalier Noir
    "bB",     # 9 : Fou Noir
    "bR",     # 10 : Tour Noire
    "bQ",     # 11 : Dame Noire
    "bK",     # 12 : Roi Noir
]
```

> **Astuce importante pour les caméras légèrement inclinées (`CTX_PAD = 8px`)** :  
> Lorsqu'on découpe une case `(col, row)` dans l'image `warped` 512×512 (où chaque case fait 64×64 px), le haut d'une grande pièce (Roi, Dame, Fou) peut dépasser légèrement de quelques pixels hors de sa case.  
> Au lieu de découper strictement `64×64`, on extrait une vignette de `80×80` pixels centrée sur la case (soit `64 + 2*8` pixels avec `cv2.copyMakeBorder` sur les bords du plateau), puis on la redimensionne en `96×96` pour `MobileNetV3`. Ainsi, le réseau voit à la fois la base ET la couronne/tête de la pièce !

---

## 3. Plan d'Implémentation Détaillé (Fichier par Fichier)

### Tâche 1 : Créer `dataset_collector.py` (`DatasetCollector`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\dataset_collector.py` :
1. **Extraction standardisée des 64 vignettes (`extract_64_patches(warped_512, pad=8, out_size=96)`)** :
   - Ajoute une bordure de `pad=8` pixels autour de `warped_512` (`cv2.copyMakeBorder(..., cv2.BORDER_REPLICATE)`).
   - Pour chaque case `sq` (`0..63`, où `col = sq % 8`, `row = 7 - (sq // 8)`) :
     - Extrait `[row*64 : (row+1)*64 + 2*pad, col*64 : (col+1)*64 + 2*pad]` (taille `80×80`).
     - Redimensionne en `96×96` (`cv2.resize(..., (96, 96), interpolation=cv2.INTER_AREA)`).
   - Retourne le dictionnaire `{sq: patch_bgr_96x96}`.
2. **Sauvegarde auto-labellisée (`save_labeled_board(warped_512, chess_board)`)** :
   - Parcourt `sq` de `0` à `63`.
   - Lit `piece = chess_board.piece_at(sq)` :
     - Si `piece is None` $\rightarrow$ `label = "empty"`
     - Sinon $\rightarrow$ `label = ("w" if piece.color == chess.WHITE else "b") + piece.symbol().upper()`
   - Enregistre l'image dans `dataset/<label>/snap_<timestamp>_<square_name>.jpg`.
   - Option d'équilibrage intelligent : comme chaque plateau contient 32 à 50 cases `empty`, n'enregistrer aléatoirement que 25% des cases `empty` (mais 100% des pièces et surtout 2 copies légèrement décalées des pièces rares : `wK, wQ, bK, bQ`) pour éviter que la classe `empty` n'écrase le dataset.
3. **Statistiques du Dataset (`get_dataset_stats()`)** :
   - Compte le nombre d'images présentes dans chacun des 13 sous-dossiers `dataset/<class_name>/` et indique si le dataset est prêt pour l'entraînement (recommandé : $\ge 10$ exemples pour chaque Roi/Dame, soit ~8 à 10 captures de plateau).

---

### Tâche 2 : Créer le Script de Fine-Tuning GPU `train_piece_classifier.py`

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\train_piece_classifier.py` (exécutable en ligne de commande `python train_piece_classifier.py --epochs 20` ou lançable depuis l'UI Web) :

1. **Architecture Modèle** :
   - Utilise `torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)`.
   - Remplace `model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 13)`.
   - Détecte automatiquement `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
2. **Data Augmentation (Crucial pour la robustesse dans le salon)** :
   - `RandomResizedCrop(96, scale=(0.88, 1.0))` (rend le modèle insensible à une petite erreur de placement des 4 coins !).
   - `ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05)` (rend le modèle robuste aux changements d'éclairage jour/soir du salon !).
   - `RandomRotation(degrees=6)`.
   - Normalisation ImageNet standard : `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
3. **Gestion du déséquilibre des classes (`Class Weights`)** :
   - Calcule l'effectif $N_c$ de chaque classe $c \in [0..12]$.
   - Calcule le poids inverse $w_c = \frac{\sum N_i}{13 \cdot \max(N_c, 1)}$ passé à `nn.CrossEntropyLoss(weight=weights)` pour que le Roi et la Dame (1 seul exemplaire par plateau) aient autant d'importance que les Pions (8 par plateau).
4. **Entraînement & Export ONNX automatique** :
   - Optimiseur `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR`.
   - Sauvegarde les poids PyTorch dans `models/chess_mobilenetv3.pt`.
   - Exporte en ONNX (`torch.onnx.export`) dans **`models/chess_mobilenetv3.onnx`** avec un axe batch dynamique ou fixe `(64, 3, 96, 96)` et `opset_version=13`.
   - Écrit les métriques finales (`val_accuracy`, date d'entraînement, nombre d'images) dans `models/model_metadata.json`.

---

### Tâche 3 : Créer le Moteur d'Inférence CPU `onnx_classifier.py` (`OnnxPieceClassifier`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\onnx_classifier.py` :
1. **Zéro dépendance PyTorch requise à l'exécution** :
   - Charge `models/chess_mobilenetv3.onnx` via **`cv2.dnn.readNetFromONNX("models/chess_mobilenetv3.onnx")`** (ou `onnxruntime.InferenceSession` si installé).
2. **Inférence Batch des 64 cases en une seule passe (`predict_board_fen(warped_512)`)** :
   - Appelle `extract_64_patches(warped_512)` pour obtenir les 64 images `96×96` BGR.
   - Construit le blob batché `blob = cv2.dnn.blobFromImages(patches_rgb, scalefactor=1.0/255.0, size=(96, 96))` puis applique la normalisation `(blob - mean) / std`.
   - Exécute `net.setInput(blob)` et `logits = net.forward()` $\rightarrow$ matrice de taille `(64, 13)` en **~30-50 ms sur CPU**.
   - Applique `softmax` pour obtenir la matrice de probabilités `probs[64, 13]`.
3. **Post-Traitement par Contraintes Légales d'Échecs (`_apply_chess_constraints(probs)`)** :
   - **Règle 1 (Pions interdits sur les rangées 1 et 8)** : Pour toutes les cases des rangées 1 et 8 (`sq < 8` ou `sq >= 56`), mettre `probs[sq, wP] = 0` et `probs[sq, bP] = 0`.
   - **Règle 2 (Exactement 1 Roi Blanc et 1 Roi Noir)** :
     - Choisir la case `sq_wK = argmax(probs[:, wK])` comme unique Roi Blanc, et mettre `probs[other_sq, wK] = 0` ailleurs.
     - Choisir la case `sq_bK = argmax(probs[:, bK])` comme unique Roi Noir, et mettre `probs[other_sq, bK] = 0` ailleurs.
   - Construire l'objet `chess.Board()` correspondant et retourner :
     - `fen` : la chaîne FEN complète (ex: `"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"`).
     - `confidence` : la probabilité moyenne sur les 64 cases.
     - `low_confidence_squares` : liste des cases où le modèle hésite (`prob < 0.70`), affichées avec un contour jaune dans l'UI pour que le joueur puisse vérifier d'un coup d'œil !

---

### Tâche 4 : Intégration dans `web_app.py` et l'Interface Web

1. **Nouvelles Routes API** :
   - `GET /api/ml/status` : retourne les statistiques du dataset (`dataset_collector.get_dataset_stats()`) et si le modèle `models/chess_mobilenetv3.onnx` est prêt (`model_metadata.json`).
   - `POST /api/ml/capture-sample` : découpe les 64 cases actuelles et les ajoute dans `dataset/` selon l'état actuel du plateau (`chess_game.board`).
   - `POST /api/ml/scan-position` : exécute `onnx_classifier.predict_board_fen(warped_512)`, met à jour `chess_game.board` avec le FEN détecté, synchronise `reference_warped` et permet de **démarrer ou reprendre immédiatement la partie depuis cette position de milieu de jeu** !
2. **Nouvel Onglet / Carte dans l'UI Web : `"🧠 IA Milieu de Partie (MobileNetV3 ONNX)"`** :
   - **Bouton principal en cours de jeu** : **`🔍 Scanner Position Quelconque (Milieu de Partie)`** + sélecteur `"Trait aux Blancs / Trait aux Noirs"` + possibilité de cliquer sur une case de l'échiquier 2D pour corriger une pièce si besoin.
   - **Section Collecte & Entraînement** :
     - Compteur d'images par classe (`Pions: 120, Cavaliers: 32, Rois: 16...`).
     - Bouton **`📸 Capturer Plateau vers Dataset`** + case à cocher **`Auto-capturer à chaque coup joué`**.
     - Commande prête à copier (ou bouton direct) pour lancer `python train_piece_classifier.py` sur le GPU.

---

## 4. Vérification & Tests d'Acceptation (Slice 5)

1. **Test Auto-Collecte** :
   - Poser les pièces en position initiale, cliquer 5 fois sur `📸 Capturer Plateau vers Dataset` en variant légèrement l'angle ou la lumière : vérifier que les 13 sous-dossiers de `dataset/` contiennent bien les vignettes `96×96` correctement triées.
2. **Test Fine-Tuning & Export ONNX** :
   - Lancer `python train_piece_classifier.py --epochs 15` : vérifier que `models/chess_mobilenetv3.onnx` est créé et que la précision de validation dépasse 95%.
3. **Test Scan Milieu de Partie en 100% CPU** :
   - Placer une position de milieu de partie aléatoire sur l'échiquier physique, cliquer sur **`🔍 Scanner Position Quelconque`** : vérifier que l'inférence ONNX via `cv2.dnn` s'exécute en `< 100 ms` sur CPU et met à jour l'échiquier avec le bon FEN, puis que l'Approche B (Slice 2) et le Bot AR (Slice 3) prennent immédiatement le relais pour jouer la suite de la partie !
