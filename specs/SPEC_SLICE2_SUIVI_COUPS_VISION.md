# Spec & Plan d'Implémentation — Slice 2 : Détection de Stabilité & Suivi des Coups par Différence (Approche B)

**Date** : 2026-10-30  
**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Prérequis** : Slice 1 terminée (`camera.py`, `chess_vision.py`, `web_app.py`)  
**Objectif CPU** : 100% OpenCV + NumPy + `python-chess` (Zéro GPU, compatible Raspberry Pi)

---

## 1. Objectif & Contexte

Une fois que l'échiquier est redressé en une image `warped` de 512×512 pixels (64 cases de 64×64 px) grâce à la Slice 1, cette **Slice 2** implémente l'**Approche B (Suivi d'état par différence croisé avec les coups légaux `python-chess`)**.

Pourquoi cette approche est idéale :
- Pas besoin d'un lourd réseau de neurones (YOLO/ResNet) pour différencier un Fou d'un Pion sous un éclairage de salon.
- On connaît la position de départ standard (ou une position FEN initiale).
- À chaque tour, `python-chess` fournit la liste exacte des ~20 à 40 **coups légaux possibles**.
- En observant **quelles cases ont changé d'apparence** entre la photo de référence avant le coup (`reference_warped`) et la nouvelle photo stable après que la main s'est retirée (`current_stable_warped`), on identifie de manière déterministe le coup joué !

Fonctionnalités livrées dans cette Slice 2 :
1. **Calque AR de Vérification du Placement Initial (`SETUP_GUIDE`)** :
   - Au lancement d'une partie, projette sur le flux AR les zones où doivent se trouver les pièces blanches (rangées 1-2) et noires (rangées 7-8) + indique si les 64 cases ont bien le profil attendu (cases centrales 3-6 vides, rangées 1-2 et 7-8 occupées).
2. **Détecteur de Mouvement / Main (`MotionGate`)** :
   - Compare chaque frame `warped` 512×512 avec la frame précédente.
   - Tant qu'une main (ou le futur bras robot) bouge au-dessus du plateau, l'état visuel passe à `"🖐️ Mouvement détecté..."` et bloque toute détection de coup.
   - Dès que l'image redevient stable pendant `STABILITY_DURATION = 0.75s`, déclenche l'analyse du coup.
3. **Algorithme de Scoring des Coups Légaux (`chess_game.py` + `chess_vision.py`)** :
   - Gère les coups normaux, les **prises (captures)**, le **roque** (4 cases modifiées) et la **prise en passant** (3 cases modifiées).
   - Met à jour le plateau `chess.Board()` automatiquement ou avec un mode confirmation, et affiche l'historique PGN/UCI + un échiquier 2D numérique synchronisé dans l'UI Web.

---

## 2. Architecture & Algorithme de Détection (Slice 2)

```text
Image Warped 512x512 (t) ──► [1. MotionGate (Diff t vs t-1)]
                                   │
                    (Stable depuis >= 0.75 seconde ?)
                                   │ OUI
                                   ▼
                 [2. Différence par Case (64 cases 64x64)]
                 Comparaison entre `reference_warped` (état début du tour)
                 et `current_stable_warped` (état stable actuel)
                                   │
                                   ▼
                 [3. Score de changement S(sq) pour chacune des 64 cases]
                 Focus sur le "cœur" de la case (40x40 central) pour ignorer
                 les débordements dus à l'angle de la caméra !
                                   │
                                   ▼
                 [4. Croisement avec `board.legal_moves` (python-chess)]
                 Pour chaque coup légal `m`, calcul du score de vraisemblance
                 selon les cases touchées (départ, arrivée, capture, roque)
```

---

## 3. Plan d'Implémentation Détaillé (Fichier par Fichier)

### Tâche 1 : Enrichir `chess_vision.py` avec l'analyse des 64 cases

Ajouter dans `ChessVisionManager` (`chess_vision.py`) les méthodes d'extraction et de comparaison :

#### 1. Extraction de la ROI interne d'une case (`get_square_roi(warped_img, col, row, margin=12)`)
- **Pourquoi `margin=12` ?** Comme la caméra n'est pas forcément pile à la verticale (90°), le sommet d'une pièce peut déborder sur le bord de la case voisine, et les lignes de la grille peuvent légèrement bouger. En prenant le carré central de `40×40` pixels (`[12:52, 12:52]` dans chaque case de `64×64`), on cible exactement la **base de la pièce** posée sur la case !
- `col` $\in [0..7]$ (`a..h`), `row` $\in [0..7]$ (`8..1`, donc `row = 7 - chess.square_rank(sq)` et `col = chess.square_file(sq)`).

#### 2. Détecteur de stabilité (`check_frame_stability(warped_frame)`)
- Convertir `warped_frame` en niveaux de gris + flou gaussien `5×5`.
- Calculer `diff = cv2.absdiff(gray, self._prev_gray)`.
- `motion_ratio = np.mean(diff > 18) ` (pourcentage de pixels ayant changé significativement).
- Si `motion_ratio > 0.008` (main présente ou en mouvement) :
  - `self.is_board_stable = False`
  - `self.last_motion_time = time.time()`
- Sinon, si `time.time() - self.last_motion_time >= self.stability_delay` (par défaut `0.75` sec) :
  - `self.is_board_stable = True`

#### 3. Carte des changements des 64 cases (`compute_square_changes(ref_warped, cur_warped)`)
Pour chaque case `sq` (`0..63` au sens `python-chess`, où `a1=0, h8=63`) :
- Extraire la ROI `40×40` dans `ref_warped` et `cur_warped`.
- Convertir en espace **LAB** (`cv2.cvtColor(..., cv2.COLOR_BGR2LAB)`) et **Gris** :
  - `diff_l = np.mean(np.abs(cur_lab[:,:,0].astype(float) - ref_lab[:,:,0].astype(float)))` (changement de luminance)
  - `diff_ab = np.mean(np.abs(cur_lab[:,:,1:].astype(float) - ref_lab[:,:,1:].astype(float)))` (changement de teinte/couleur)
  - `edge_ref = cv2.Laplacian(ref_gray, cv2.CV_64F).var()` et `edge_cur = cv2.Laplacian(cur_gray, cv2.CV_64F).var()` (texture/contraste : une case occupée par une pièce 3D présente beaucoup plus de gradients/bords et de variance qu'une case plate vide).
- Score global de changement de la case `sq` :
  - `change_score[sq] = 0.6 * diff_l + 0.8 * diff_ab + 0.15 * abs(edge_cur - edge_ref)`
- Estimer aussi la polarité de couleur de l'occupant actuel (`estimate_piece_polarity(cur_warped, sq)`) :
  - Compare la luminance moyenne du centre `32×32` de la case avec la luminance de référence des cases blanches/noires pour aider à départager une **prise (capture)** (une pièce blanche remplace une pièce noire sur la case d'arrivée : la case était déjà occupée, mais sa couleur passe du sombre au clair !).

---

### Tâche 2 : Créer `chess_game.py` (`ChessGameManager`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\chess_game.py` qui encapsule `chess.Board()` et la machine à états de la partie :

#### États de la partie (`self.state`) :
- `"IDLE"` : Partie non démarrée (mode calibration libre).
- `"SETUP_CHECK"` : Guide AR affiché pour poser les 32 pièces sur les rangées 1-2 et 7-8 avant de capturer l'image de référence initiale.
- `"WAIT_HUMAN_MOVE"` : C'est au tour du joueur humain ; le système surveille les changements stables par rapport à `self.reference_warped`.
- `"WAIT_BOT_THINKING"` : (Utilisé en Slice 3) Le moteur calcule son coup.
- `"WAIT_BOT_EXECUTION"` : (Utilisé en Slice 3) Le coup du bot est affiché en AR ; le système attend que l'humain (ou le robot SO-101) déplace physiquement la pièce du bot sur le plateau.
- `"GAME_OVER"` : Échec et mat, pat ou nulle.

#### Algorithme de détection du coup légal (`detect_played_move(self, change_scores, cur_warped)`) :
1. Vérifier qu'au moins 2 cases dépassent le seuil minimal de changement `MIN_CHANGE_THRESHOLD = 14.0`. Sinon, aucun coup n'a encore été joué.
2. Pour chaque coup `move` dans `self.board.legal_moves` :
   - Déterminer l'ensemble des **cases attendues en changement (`expected_squares`)** :
     - **Roque (`board.is_castling(move)`)** : 4 cases (`move.from_square`, `move.to_square`, ainsi que la case de départ et d'arrivée de la Tour !).
     - **Prise en passant (`board.is_en_passant(move)`)** : 3 cases (`from_square`, `to_square`, et la case du pion capturé).
     - **Coup standard ou Capture normale** : 2 cases (`move.from_square`, `move.to_square`).
   - Calculer le score du coup `move` :
     - `pos_score = mean([change_scores[sq] for sq in expected_squares])`
     - `noise_score = max([change_scores[sq] for sq in all_other_squares])`
     - Si toutes les cases `expected_squares` ont un `change_score > MIN_CHANGE_THRESHOLD` :
       - `move_score = pos_score - 0.7 * noise_score`
     - Bonus si `board.is_castling(move)` et que les 4 cases dépassent toutes le seuil (pour éviter de confondre un roque avec un simple déplacement de Roi).
3. Si le meilleur coup `best_move` a un `move_score` supérieur au seuil de confiance et reste stable pendant 2 analyses consécutives (~0.5s) :
   - Jouer `self.board.push(best_move)`.
   - Mettre à jour `self.reference_warped = cur_warped.copy()` !
   - Enregistrer le coup dans `self.move_history` avec son format SAN (`e4`, `Nf3`, `O-O`) et UCI (`e2e4`).

#### Méthode de secours & Correction manuelle (Indispensable en conditions réelles !) :
- **`sync_reference_frame(cur_warped)`** : Si l'éclairage du salon change brusquement ou qu'on ajuste une pièce mal centrée ("J'adoube"), un bouton **"📸 Recalibrer l'image de référence"** met à jour `self.reference_warped = cur_warped.copy()` sans changer l'état logique `chess.Board`.
- **`manual_push_uci(uci_str, cur_warped)`** : Permet de jouer/forcer un coup depuis l'interface Web si besoin, tout en resynchronisant `reference_warped`.
- **`undo_last_move(cur_warped)`** : Annule le dernier coup (`board.pop()`) et resynchronise `reference_warped`.

---

### Tâche 3 : Overlays AR de la Slice 2 dans `chess_vision.py`

Ajouter dans `ChessVisionManager.process_frame()` le dessin en perspective (via `self.H_inv` et `cv2.addWeighted` pour la transparence) :
1. **En mode `"SETUP_CHECK"`** :
   - Colorer en vert translucide les rangées 1 et 2 (cases `a1..h2`, pièces blanches) et en bleu translucide les rangées 7 et 8 (cases `a7..h8`, pièces noires) directement sur le vrai plateau pour guider l'installation.
2. **En cours de partie** :
   - **Dernier coup joué** : Colorer en bleu/cyan semi-transparent (`alpha = 0.35`) la case de départ et la case d'arrivée du dernier coup validé.
   - **Roi en échec** : Si `board.is_check()`, colorer la case du Roi menacé en **Rouge vif clignotant**.
   - **Indicateur de stabilité en haut de l'image AR** : Badge vert `"● PLATEAU STABLE"` ou orange `"🖐️ MOUVEMENT EN COURS..."` + affichage en temps réel des 2 cases présentant le plus fort changement (`Top diff: e2 (42.1), e4 (38.7)`) pour faciliter le diagnostic.

---

### Tâche 4 : Nouvelles Routes API dans `web_app.py` & Interface Web

1. **Routes API Partie (`/api/game/...`)** :
   - `GET /api/game/status` : retourne `state`, `fen`, `turn` (`"white"` / `"black"`), `is_check`, `is_game_over`, `result`, `move_history` (liste `{move_number, white_san, black_san, uci}`), `is_board_stable`, `top_changed_squares`, `last_move`.
   - `POST /api/game/new` : passe en mode `"SETUP_CHECK"` (ou démarre directement la partie en capturant `reference_warped`).
   - `POST /api/game/confirm-setup` : capture `reference_warped = vision_manager.get_latest_warped_frame()` et passe à `"WAIT_HUMAN_MOVE"`.
   - `POST /api/game/resync-reference` : recalibre `reference_warped` sur la frame actuelle ("J'adoube" / changement de lumière).
   - `POST /api/game/manual-move` : joue un coup UCI manuellement (`{"uci": "e2e4"}`) et resynchronise `reference_warped`.
   - `POST /api/game/undo` : annule le dernier coup.
2. **Ajouts dans l'UI Web (`templates/index.html` & `static/js/app.js`)** :
   - **Mini-échiquier 2D numérique interactif** (rendu en HTML/CSS ou SVG avec les pièces Unicode `♔♕♖♗♘♙`) synchronisé avec le FEN actuel.
   - **Boutons de contrôle de partie** : `🆕 Nouvelle Partie`, `✅ Valider Plateau Initial`, `📸 Recalibrer Référence (J'adoube)`, `↩️ Annuler Coup`.
   - **Liste des coups (PGN)** et indicateur visuel de stabilité (`Plateau Stable` vs `Mouvement détecté`).

---

## 4. Vérification & Tests d'Acceptation (Slice 2)

1. **Test unitaire de détection de coup sur images synthétiques** :
   - Générer une image 512×512 simulant la position initiale (pions et pièces dessinés par des disques clairs/foncés sur les rangées 1-2 et 7-8).
   - Simuler le déplacement du pion `e2` vers `e4` (et un second test avec le petit roque `e1g1` + `h1f1`).
   - Vérifier que `compute_square_changes()` + `detect_played_move()` identifie `e2e4` en premier choix avec une marge nette, et `e1g1` pour le roque.
2. **Test en direct avec la Webcam** :
   - Cliquer sur `✅ Valider Plateau Initial`, bouger un pion à la main : vérifier que pendant que la main est au-dessus du plateau, le badge affiche `"🖐️ Mouvement en cours..."`, puis ~0.8s après le retrait de la main, le coup `e2e4` s'ajoute dans l'historique et s'illumine en AR sur le plateau.
