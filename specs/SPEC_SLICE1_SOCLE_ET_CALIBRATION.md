# Spec & Plan d'Implémentation — Slice 1 : Socle Web, Caméra & Calibration Perspective ($H / H^{-1}$)

**Date** : 2026-10-30  
**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Dépendances Python** : `flask`, `flask-cors`, `opencv-python`, `numpy`, `chess` (`pip install chess`)  
**Serveur Web** : Flask sur `http://127.0.0.1:5001` (`start_web.py`)

---

## 1. Objectif & Contexte

Cette **Slice 1** construit le socle complet du projet `Chess_AR_Vision` et résout le problème géométrique fondamental : **comment localiser un échiquier filmé par une webcam (vue de dessus ou en biais), redresser sa perspective en une grille plate 512×512 (`warp`), et re-projeter des dessins en Réalité Augmentée sur le flux caméra réel (`inverse warp`)**.

Fonctionnalités livrées dans cette Slice 1 :
1. **Acquisition Webcam threadée (`camera.py`)** : Architecture identique à `Robot_Arm_SO-101` avec hook `frame_processor`, rotation 180°, miroir horizontal et snapshot.
2. **Calibration Hybride de l'Échiquier (`chess_vision.py`)** :
   - **Mode Auto** : Détection automatique des 4 coins extérieurs de la grille 8×8 via OpenCV (`findChessboardCornersSB` / `findChessboardCorners` sur les 7×7 coins internes puis extrapolation mathématique d'une case vers l'extérieur, avec fallback par détection du plus grand quadrilatère de contour).
   - **Mode Manuel / Édition interactive dans le navigateur** : Affichage des 4 coins (`A8`, `H8`, `H1`, `A1`) directement par-dessus la vidéo. L'utilisateur peut cliquer 4 coins *from scratch* ou **glisser-déposer (drag & drop)** n'importe quel coin pour ajuster au pixel près la détection automatique.
   - **Rotation de l'orientation (90°)** : Bouton permettant de faire tourner l'assignation `[A8, H8, H1, A1]` d'un quart de tour pour choisir de quel côté se trouvent les Blancs (rangée 1).
   - **Persistance JSON** : Sauvegarde et rechargement automatique dans `board_calibration.json`.
3. **Double Flux Vidéo MJPEG** :
   - **Flux Principal AR (`/api/camera/stream`)** : Image de la caméra avec la grille 8×8 en Réalité Augmentée collée à la perspective réelle du plateau et les coordonnées `a1..h8`.
   - **Flux Miniature 2D Redressé (`/api/vision/warped_stream`)** : Image carrée 512×512 vue de dessus après `cv2.warpPerspective`, découpée en 64 cases de 64×64 pixels.

---

## 2. Architecture Cible (Slice 1)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         camera.py (CameraManager)                           │
│  - Thread daemon d'acquisition OpenCV (CAP_DSHOW sous Windows)              │
│  - Appelle `self.frame_processor(frame)` à chaque frame                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    chess_vision.py (ChessVisionManager)                     │
│  - Gère les 4 coins [A8, H8, H1, A1] en coordonnées normalisées [0.0..1.0]  │
│  - Calcule `H` (Caméra -> 512x512) et `H_inv` (512x512 -> Caméra)           │
│  - `auto_detect_board(raw_frame)` : 7x7 interne -> extrapolation 8x8        │
│  - `process_frame(frame)` :                                                 │
│      1. Génère `self.latest_warped_jpeg` (vue plate 512x512)                │
│      2. Dessine l'overlay AR (grille perspective + labels a1..h8) via H_inv │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                web_app.py                                   │
│  - Routes Caméra : /api/camera/list, connect, disconnect, settings, stream  │
│  - Routes Vision :                                                          │
│      GET  /api/vision/status          -> Coins, matrices, options AR        │
│      POST /api/vision/auto-calibrate  -> Lance auto_detect_board()          │
│      POST /api/vision/corners         -> Met à jour les 4 coins [A8,H8,H1,A1]│
│      POST /api/vision/rotate-board    -> Décale circulairement les 4 coins  │
│      POST /api/vision/overlay-settings-> Active/désactive grille, labels... │
│      GET  /api/vision/warped_stream   -> Flux MJPEG 512x512 redressé        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Plan d'Implémentation Détaillé (Fichier par Fichier)

### Tâche 1 : Créer `requirements.txt`, `start_web.py` et `LANCER_SERVEUR.bat`

1. **`requirements.txt`** :
   ```text
   Flask==3.0.0
   flask-cors==4.0.0
   opencv-python>=4.8.0
   numpy>=1.24.0
   chess>=1.10.0
   ```
2. **`start_web.py`** :
   - Lance l'application Flask de `web_app.py` sur `host="0.0.0.0"`, `port=5001`, `debug=False`, `threaded=True`.
   - Ouvre automatiquement `http://127.0.0.1:5001` dans le navigateur après 1 seconde.
3. **`LANCER_SERVEUR.bat`** :
   - Script Windows lançant `python start_web.py`.

---

### Tâche 2 : Créer `camera.py` (`CameraManager`)

Reprendre la structure éprouvée de `CameraManager` dans `Robot_Arm_SO-101` avec :
- `self.width = 640`, `self.height = 480` (ou 1280×720 en option).
- `self.show_crosshair = False` par défaut (car ici l'overlay principal est la grille d'échecs AR).
- `self.rotate_180 = False`, `self.flip_h = False`.
- `self.frame_processor = None` : appelé à chaque frame dans `_capture_loop` : `display_frame = self.frame_processor(raw_frame_copy)`.
- Méthode `get_raw_frame()` : retourne une copie thread-safe de `self._latest_raw_frame` (indispensable pour lancer la calibration automatique sur une image vierge sans overlay).

---

### Tâche 3 : Créer `chess_vision.py` (`ChessVisionManager`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\chess_vision.py` :

#### Convention géométrique de référence (Repère 2D redressé 512×512)
- Taille de l'image redressée : `WARP_SIZE = 512` (soit `SQUARE_SIZE = 64` pixels par case).
- Ordre strict des 4 coins (`self.corners_norm`) : liste de 4 points `[x_norm, y_norm]` compris entre `0.0` et `1.0` (pour être indépendants de la résolution de la caméra) dans l'ordre horaire en partant du coin **Haut-Gauche (`A8`)** :
  1. Index `0` : Coin extérieur **A8** $\rightarrow$ correspond à `(0, 0)` dans l'image 512×512.
  2. Index `1` : Coin extérieur **H8** $\rightarrow$ correspond à `(512, 0)` dans l'image 512×512.
  3. Index `2` : Coin extérieur **H1** $\rightarrow$ correspond à `(512, 512)` dans l'image 512×512.
  4. Index `3` : Coin extérieur **A1** $\rightarrow$ correspond à `(0, 512)` dans l'image 512×512.
- Ainsi, dans la vue 512×512 redressée :
  - La colonne `col` (`0` = colonne `a`, ..., `7` = colonne `h`) va de gauche à droite : `x1 = col * 64`, `x2 = (col + 1) * 64`.
  - La ligne `row` (`0` = rangée `8` noire en haut, ..., `7` = rangée `1` blanche en bas) va de haut en bas : `y1 = row * 64`, `y2 = (row + 1) * 64`.

#### Méthodes à implémenter dans `ChessVisionManager` :
1. **`__init__(self, calibration_file="board_calibration.json", log_fn=None)`** :
   - Charge `board_calibration.json` s'il existe, sinon initialise 4 coins par défaut centrés `[[0.2, 0.15], [0.8, 0.15], [0.85, 0.85], [0.15, 0.85]]`.
   - Booléens d'affichage AR :
     - `self.show_ar_grid = True`
     - `self.show_ar_labels = True`
     - `self.show_warped_grid = True`
   - Matrices mises en cache : `self.H = None`, `self.H_inv = None`.
   - `self._latest_warped_frame = None` (image 512×512 pure sans dessins, pour l'analyse des coups en Slice 2).
   - `self._latest_warped_jpeg = None` (bytes JPEG 512×512 avec grille pour `/api/vision/warped_stream`).

2. **`_compute_homography(self, frame_w, frame_h)`** :
   - Convertit `self.corners_norm` en coordonnées pixels `src_pts = np.float32([[x * frame_w, y * frame_h] for x, y in self.corners_norm])`.
   - `dst_pts = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])`.
   - `self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)`
   - `self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)`

3. **`set_corners(self, corners_norm)` & `rotate_board_90(self)`** :
   - `set_corners` valide qu'on reçoit 4 points `[x, y]` dans `[0..1]`, met à jour `self.corners_norm`, sauvegarde dans `board_calibration.json`.
   - `rotate_board_90` fait tourner circulairement la liste des 4 coins : `self.corners_norm = self.corners_norm[1:] + self.corners_norm[:1]` et sauvegarde.

4. **`auto_detect_board(self, raw_frame)` (Mode Calibration Automatique)** :
   - **Stratégie 1 (Grille interne 7×7 d'un échiquier vide ou peu chargé)** :
     - Convertir `raw_frame` en niveaux de gris + `cv2.equalizeHist`.
     - Appeler `found, corners = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)` (avec fallback `cv2.findChessboardCorners(gray, (7, 7))`).
     - Si trouvé : `corners` contient les 49 intersections internes (du coin interne entre `a8/b7` au coin interne entre `g2/h1`).
     - Calculer l'homographie entre la grille interne `dst_inner` (coordonnées `(64, 64)` à `(448, 448)` dans le repère 512×512) et les points détectés dans l'image, puis projeter les 4 coins extérieurs `(0,0), (512,0), (512,512), (0,512)` vers l'image caméra ! Cela donne les 4 coins extérieurs exacts de l'échiquier, même en forte perspective !
   - **Stratégie 2 (Fallback Contours du Plateau si des pièces sont déjà posées dessus)** :
     - Flou gaussien + `cv2.Canny` + dilatation + `cv2.findContours`.
     - Trier les contours par aire décroissante, approximer par `cv2.approxPolyDP(cnt, 0.02 * peri, True)`.
     - Prendre le plus grand quadrilatère convexe à 4 sommets couvrant au moins 15% de l'image, ordonner ses 4 sommets dans le sens horaire `[Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche]`.
   - Si l'une des stratégies réussit : normaliser par `(w, h)`, sauvegarder dans `self.corners_norm` et retourner `(True, method_name, self.corners_norm)`. Sinon retourner `(False, "Aucun échiquier détecté automatiquement. Utilisez le placement manuel des 4 coins.", None)`.

5. **`square_to_warped_xy(self, col, row)` et `warped_points_to_camera(self, pts_2d)`** :
   - Fonctions utilitaires permettant de convertir n'importe quelle case `(col, row)` ou point `(x, y)` du repère 512×512 en coordonnées pixels `(u, v)` sur l'image de la caméra via `cv2.perspectiveTransform(pts, self.H_inv)`.

6. **`process_frame(self, frame)` (Hook branché sur `CameraManager.frame_processor`)** :
   - Calcule `self._compute_homography(w, h)`.
   - Génère la vue redressée : `warped = cv2.warpPerspective(frame, self.H, (512, 512))`.
   - Stocke `self._latest_warped_frame = warped.copy()` sous lock.
   - Crée `warped_display = warped.copy()` : si `self.show_warped_grid`, dessine les 64 cases et les noms `a8..h1` dessus, puis encode en JPEG dans `self._latest_warped_jpeg`.
   - Sur `display_frame = frame.copy()` :
     - Si `self.show_ar_grid` : projette les 9 lignes horizontales et 9 lignes verticales de la grille `(0..512, pas de 64)` sur `display_frame` via `self.H_inv` avec des lignes anti-aliasées (`cv2.LINE_AA`).
     - Dessine le cadre extérieur avec un code couleur clair pour l'orientation :
       - Bordure **Rangée 1 (Côté Blancs, entre A1 et H1)** en **Vert vif** (`(80, 230, 100)`) avec le label `"COTE BLANCS (Rang 1)"`.
       - Bordure **Rangée 8 (Côté Noirs, entre A8 et H8)** en **Violet/Bleu** (`(255, 140, 80)`).
     - Si `self.show_ar_labels` : projette et affiche les coordonnées `a..h` et `1..8` le long du plateau en perspective.
   - Retourne `display_frame`.

---

### Tâche 4 : Créer `web_app.py`

Créer le serveur Flask avec :
- `GET /` : sert `templates/index.html`.
- Routes Caméra : `/api/camera/list`, `/api/camera/status`, `/api/camera/connect`, `/api/camera/disconnect`, `/api/camera/settings`, `/api/camera/stream`, `/api/camera/snapshot`.
- Routes Vision & Calibration :
  - `GET /api/vision/status` : retourne `corners_norm`, `is_calibrated`, `settings` (`show_ar_grid`, `show_ar_labels`, `show_warped_grid`).
  - `POST /api/vision/auto-calibrate` : récupère `raw_frame = camera_manager.get_raw_frame()`, appelle `vision_manager.auto_detect_board(raw_frame)` et renvoie les nouveaux coins.
  - `POST /api/vision/corners` : reçoit `{"corners": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]}` et met à jour `vision_manager.set_corners(...)`.
  - `POST /api/vision/rotate-board` : appelle `vision_manager.rotate_board_90()` pour tourner l'orientation du plateau de 90°.
  - `POST /api/vision/overlay-settings` : met à jour `show_ar_grid`, `show_ar_labels`, `show_warped_grid`.
  - `GET /api/vision/warped_stream` : flux MJPEG de la miniature 512×512 redressée (`vision_manager.get_warped_jpeg()`).

---

### Tâche 5 : Interface Web (`templates/index.html`, `static/css/style.css`, `static/js/app.js`)

Construire une interface sombre moderne (style Lichess / Dashboard Robotique) organisée en 2 colonnes :
1. **Colonne Principale (Gauche - 65%) : Vue Caméra Réalité Augmentée & Calibration Interactive**
   - Barre d'outils Caméra : Sélection webcam, Activer/Arrêter, Rotation 180°, Miroir, Snapshot.
   - Barre d'outils Calibration :
     - Bouton **`✨ Détection Auto`** (lance `/api/vision/auto-calibrate`).
     - Bouton Toggle **`📐 Éditer les 4 Coins (Manuel)`** : affiche un `<canvas>` ou `<svg>` superposé exactement par-dessus `<img id="camera-stream-img">` avec **4 poignées circulaires déplaçables à la souris (`A8`, `H8`, `H1`, `A1`)** + bouton **`🎯 Placer 4 points en cliquant`** (permet de cliquer successivement sur A8, H8, H1, A1).
     - Bouton **`🔄 Pivoter Orientation 90°`** (pour aligner le côté Blancs en un clic).
     - Toggles **`# Grille AR`** et **`🔤 Coordonnées`**.
2. **Colonne Latérale (Droite - 35%) : Vue 2D Redressée (512×512) & État**
   - Carte **"🔍 Vue Plateau Redressée (Perspective Transform 512×512)"** affichant `<img id="warped-stream-img" src="/api/vision/warped_stream">` pour vérifier visuellement que les 64 cases tombent pile dans les carrés de la grille.
   - Carte **"♟️ État de la Partie & Moteur"** (préparée pour les Slices 2 et 3) + Journal des événements en bas.

---

## 4. Vérification & Tests d'Acceptation (Slice 1)

1. **Test unitaire géométrique (sans webcam)** :
   - Créer une image synthétique 640×480 contenant un damier 8×8 dessiné en perspective trapézoïdale.
   - Vérifier que `ChessVisionManager.auto_detect_board()` retrouve les 4 coins et que `process_frame()` génère bien un `warped_jpeg` 512×512 valide.
2. **Test Serveur & Routes Flask** :
   - Lancer `start_web.py` et tester `GET /api/vision/status`, `POST /api/vision/corners`, `POST /api/vision/rotate-board`.
3. **Test Interactif Webcam** :
   - Vérifier que le déplacement d'une poignée (`A8`, `H8`, `H1`, `A1`) à la souris dans le navigateur déforme en temps réel la grille AR projetée sur la vidéo et met à jour la vue redressée 512×512.
