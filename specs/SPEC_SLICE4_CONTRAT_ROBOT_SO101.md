# Spec & Plan d'Implémentation — Slice 4 : Pont Robotique & Préparation Fusion avec `Robot_Arm_SO-101`

**Date** : 2026-10-30  
**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Projet Frère Cible** : `C:\Users\b14ko\Documents\Projets\Robot_Arm_SO-101`  
**Prérequis** : Slices 1, 2 et 3 terminées

---

## 1. Objectif & Contexte

Une fois que les Slices 1, 2 et 3 fonctionnent en **Mode Manuel AR** (où l'humain déplace la pièce du Bot en suivant la flèche AR), cette **Slice 4** prépare toute la couche de traduction physique et l'API d'interconnexion pour que le bras robotique **`Robot_Arm_SO-101`** puisse exécuter les coups automatiquement.

Pourquoi un coup d'échecs (`g1f3` ou `e4d5`) ne suffit pas à un bras robotique :
1. **Un coup de prise (`is_capture = True`) nécessite 2 mouvements physiques** :
   - Mouvement 1 : Aller chercher la pièce adverse sur la case d'arrivée (`d5`) et la déposer hors du plateau dans la **zone cimetière (`graveyard`)**.
   - Mouvement 2 : Aller chercher la pièce sur la case de départ (`e4`) et la déposer sur la case d'arrivée (`d5`).
2. **Un roque (`is_castling = True`) nécessite 2 déplacements sur le plateau** :
   - Mouvement 1 : Déplacer le Roi (`e8` $\rightarrow$ `g8`).
   - Mouvement 2 : Déplacer la Tour (`h8` $\rightarrow$ `f8`).
3. **Une prise en passant (`is_en_passant = True`)** :
   - Le pion capturé ne se trouve **pas** sur la case d'arrivée (`to_square`), mais sur la case adjacente ! Il faut l'enlever vers le cimetière après ou avant le déplacement du pion.
4. **Chaque pièce a une hauteur et un diamètre différents** :
   - Saisir un Roi (`~75-90 mm`) ou un Pion (`~40-50 mm`) demande une hauteur d'approche $Z$ et une ouverture de pince (`gripper`) adaptées.
5. **Coordonnées Spatiales des 64 cases** :
   - Le robot a besoin des coordonnées exactes `(x_norm, y_norm)` dans le repère normalisé du plateau `[-1.0 .. +1.0]` ou `[0.0 .. 1.0]` ET des coordonnées pixels `(u, v)` dans la caméra pour chaque case `a1..h8`.

---

## 2. Architecture Cible (Slice 4)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        chess_game.py (pending_bot_move)                     │
│  - Génère un coup logique python-chess (ex: `e4xd5` ou `O-O`)               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       robot_bridge.py (RobotChessBridge)                    │
│  - Traduit tout coup `chess.Move` en une liste ordonnée d'actions atomiques │
│    `PICK_AND_PLACE` avec :                                                  │
│      • Coordonnées Plateau (col, row, x_mm, y_mm)                           │
│      • Coordonnées Caméra Pixels (u, v) via `H_inv`                         │
│      • Type de pièce (`pawn`, `knight`, ..., `king`), hauteur Z & ouverture │
│      • Position de retour sécurisée (`HOME_SAFE_OUTSIDE_BOARD`)             │
│  - Mode d'exécution :                                                       │
│      1. `MANUAL_AR` : L'humain déplace la pièce (Slice 3)                   │
│      2. `HTTP_WEBHOOK_SO101` : Envoie la séquence JSON à `127.0.0.1:5000`   │
│      3. `DIRECT_MODULE` : Appel direct de `MotorController` après fusion    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Plan d'Implémentation Détaillé (Fichier par Fichier)

### Tâche 1 : Créer `robot_bridge.py` (`RobotChessBridge`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\robot_bridge.py` :

#### 1. Configuration géométrique physique du plateau (`board_physical_config`)
- `square_size_mm = 45.0` (taille standard d'une case en millimètres, configurable dans l'UI).
- `piece_profiles` (hauteur de prise et ouverture de pince SO-101 de `0` à `100%`) :
  ```python
  PIECE_PROFILES = {
      "pawn":   {"grasp_z_mm": 22.0, "lift_z_mm": 85.0, "gripper_open": 55, "gripper_close": 22},
      "knight": {"grasp_z_mm": 30.0, "lift_z_mm": 90.0, "gripper_open": 60, "gripper_close": 25},
      "bishop": {"grasp_z_mm": 34.0, "lift_z_mm": 95.0, "gripper_open": 60, "gripper_close": 24},
      "rook":   {"grasp_z_mm": 28.0, "lift_z_mm": 90.0, "gripper_open": 62, "gripper_close": 26},
      "queen":  {"grasp_z_mm": 42.0, "lift_z_mm": 105.0, "gripper_open": 65, "gripper_close": 25},
      "king":   {"grasp_z_mm": 48.0, "lift_z_mm": 110.0, "gripper_open": 68, "gripper_close": 26},
  }
  ```
- **Zone Cimetière (`graveyard_slots`)** : Coordonnées hors-plateau à droite/gauche de l'échiquier (`col = 8.8, row = 0..7`) où le robot dépose les pièces capturées sans les empiler.

#### 2. Méthode `get_square_coordinates(square_name, vision_manager)`
Pour n'importe quelle case (`"e4"`, `"g8"`, ou `"graveyard_0"`) :
- Calcule `(col, row)` dans `[0..7]`.
- Calcule le centre dans le repère 512×512 : `wx = (col + 0.5) * 64`, `wy = (row + 0.5) * 64`.
- Calcule les coordonnées millimétriques relatives au centre du plateau :
  - `x_mm = (col - 3.5) * self.square_size_mm`
  - `y_mm = (3.5 - row) * self.square_size_mm`
- Calcule les coordonnées pixels `(cam_u, cam_v)` dans l'image caméra via `vision_manager.warped_points_to_camera([(wx, wy)])`.
- Retourne :
  ```python
  {
      "square": square_name,
      "col": col,
      "row": row,
      "warped_xy": [round(wx, 1), round(wy, 1)],
      "camera_uv": [round(cam_u, 1), round(cam_v, 1)],
      "board_mm": [round(x_mm, 1), round(y_mm, 1)]
  }
  ```

#### 3. Méthode `build_robot_action_plan(board, move, vision_manager)`
Prend l'état `chess.Board` **avant** le coup et l'objet `chess.Move`, et retourne le plan complet d'exécution robotique :
- **Cas 1 : Capture normale (`board.is_capture(move)` et non prise en passant)** :
  - `step 1` : `REMOVE_CAPTURED` : Prendre la pièce adverse en `to_sq` (avec le profil de hauteur de la pièce capturée !) $\rightarrow$ Déposer dans `graveyard_slot_N`.
  - `step 2` : `MOVE_PIECE` : Prendre la pièce du bot en `from_sq` $\rightarrow$ Déposer en `to_sq`.
  - `step 3` : `RETURN_HOME` : Revenir en position `HOME` hors du champ de la caméra (indispensable pour que `MotionGate` voie le plateau libre et valide le coup !).
- **Cas 2 : Prise en passant (`board.is_en_passant(move)`)** :
  - Identifie la case réelle du pion capturé (`captured_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))`).
  - `step 1` : `REMOVE_CAPTURED` depuis `captured_sq` $\rightarrow$ `graveyard_slot_N`.
  - `step 2` : `MOVE_PIECE` depuis `from_sq` $\rightarrow$ `to_sq`.
  - `step 3` : `RETURN_HOME`.
- **Cas 3 : Roque (`board.is_castling(move)`)** :
  - `step 1` : `MOVE_KING` (`e8` $\rightarrow$ `g8` ou `c8`).
  - `step 2` : `MOVE_ROOK` (`h8` $\rightarrow$ `f8` ou `a8` $\rightarrow$ `d8`).
  - `step 3` : `RETURN_HOME`.
- **Cas 4 : Déplacement simple** :
  - `step 1` : `MOVE_PIECE` (`from_sq` $\rightarrow$ `to_sq`).
  - `step 2` : `RETURN_HOME`.

---

### Tâche 2 : Visualisation AR de la Séquence Robot et Routes API

1. **Nouvelles Routes API dans `web_app.py`** :
   - `GET /api/robot/grid-coordinates` : exporte la carte complète des 64 cases (`a1..h8`) avec leurs coordonnées `camera_uv`, `warped_xy` et `board_mm` (prêt à être consommé par le module de calibration de `Robot_Arm_SO-101`).
   - `GET /api/robot/current-plan` : retourne le JSON complet du plan d'action robotique (`build_robot_action_plan`) du coup en attente.
   - `POST /api/robot/simulate-move` : permet de tester la génération du plan robotique sur n'importe quel coup (ex: capture, roque) et de voir sa trajectoire en AR.
2. **Panneau UI "🦾 Pont Robot SO-101 (Prêt pour Fusion)"** dans `templates/index.html` :
   - Affiche en temps réel la séquence d'instructions robotiques générée pour chaque coup du Bot (ex: `1. Saisir Pion en d4 (Z=22mm) ➔ Cimetière #1 | 2. Saisir Cavalier en f6 (Z=30mm) ➔ Poser en d4 | 3. Retour Home`).
   - Bouton pour copier/inspecter la table des coordonnées des 64 cases.

---

## 4. Guide de Fusion Future avec `Robot_Arm_SO-101`

Lorsque les 4 Slices seront validées dans `Chess_AR_Vision`, la fusion dans `Robot_Arm_SO-101` se fera en 3 opérations simples :
1. Copier `chess_vision.py`, `chess_game.py`, `chess_engine.py` et `robot_bridge.py` dans `Robot_Arm_SO-101`.
2. Brancher `camera_manager.frame_processor = vision_manager.process_frame` dans le `web_app.py` de `Robot_Arm_SO-101`.
3. Relier chaque étape `PICK_AND_PLACE` de `robot_bridge.py` aux positions enregistrées / cinématique inverse du `MotorController` de `Robot_Arm_SO-101`.
