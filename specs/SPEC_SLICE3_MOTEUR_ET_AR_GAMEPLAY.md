# Spec & Plan d'Implémentation — Slice 3 : Moteur Multi-Niveaux (Stockfish + Fallback) & Gameplay AR Complet

**Date** : 2026-10-30  
**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Prérequis** : Slices 1 & 2 terminées  
**Objectif** : Jouer une partie complète contre un Bot au niveau réglable (de Débutant 600 ELO à Maître 2800+ ELO) avec affichage du coup à jouer en **Réalité Augmentée (flèche en perspective + surbrillance)** et validation automatique lorsque l'utilisateur déplace physiquement la pièce du Bot sur le plateau.

---

## 1. Objectif & Contexte

Cette **Slice 3** transforme le détecteur de coups de la Slice 2 en un **partenaire de jeu et coach en Réalité Augmentée** :
1. **Moteur d'échecs Hybride & Multi-Niveaux (`chess_engine.py`)** :
   - **Mode Stockfish UCI (si l'exécutable `stockfish` est présent sur le PC ou le Raspberry Pi)** : Utilise `chess.engine.SimpleEngine.popen_uci()` avec réglage fin de `Skill Level` (`0..20`) et `UCI_LimitStrength` + `UCI_Elo` (`1320..3000`).
   - **Mode Moteur Python Intégré (Fallback Zéro-Config immédiat & Niveaux Débutants 600–1400 ELO)** : Un moteur Minimax avec élagage Alpha-Beta, tables de positionnement des pièces (Piece-Square Tables) et injection contrôlée d'aléatoire selon l'ELO choisi. **Avantage majeur** : le projet fonctionne à 100% dès la première seconde sans rien télécharger, et offre des niveaux très accessibles (600–1200 ELO) plus naturels pour un joueur occasionnel que Stockfish.
2. **Boucle de Gameplay AR "Zéro Clavier/Souris" (`WAIT_BOT_EXECUTION`)** :
   - Dès que l'humain joue son coup sur le vrai plateau, le Bot calcule sa réponse (ex: `g8f6`).
   - Le système passe à l'état `"WAIT_BOT_EXECUTION"` : une **flèche lumineuse en perspective** part de `g8` vers `f6` sur la vidéo AR, avec l'instruction `"🤖 Coup du Bot : Cavalier g8 ➔ f6 (Déplacez la pièce sur le plateau)"`.
   - L'humain déplace la pièce du Bot sur l'échiquier réel. Dès que la caméra détecte que les cases `g8` et `f6` ont changé conformément au coup attendu par le Bot, le coup est automatiquement validé, `reference_warped` est mis à jour, la flèche s'efface et c'est de nouveau au tour de l'humain !
3. **Mode Coach / Indice AR (`Hint`) & Barre d'Évaluation** :
   - À tout moment pendant son tour, l'humain peut demander un **Indice AR** : une flèche pointillée cyan/bleue montre sur le vrai plateau le meilleur coup suggéré pour lui.
   - Une barre d'évaluation (`+1.4`, `-0.8`, `Mat en 3`) s'affiche à côté du plateau.

---

## 2. Architecture Cible (Slice 3)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        chess_engine.py (ChessEngineManager)                 │
│  - Détecte automatiquement Stockfish (dossier local `engines/`, PATH, apt)  │
│  - Fallback transparent sur le moteur Python Alpha-Beta + PST intégré       │
│  - Niveaux prédéfinis :                                                     │
│      1. Débutant (ELO ~800)   - Profondeur 1-2 + bruit 35%                  │
│      2. Facile (ELO ~1100)    - Profondeur 2 + bruit 15%                    │
│      3. Club (ELO ~1400)      - Profondeur 3 / Stockfish Skill 3            │
│      4. Avancé (ELO ~1800)    - Stockfish ELO 1800 / Alpha-Beta Prof 4      │
│      5. Expert (ELO ~2200)    - Stockfish ELO 2200                          │
│      6. Maximum (Stockfish)   - Stockfish Pleine Puissance                  │
│  - `get_best_move(board, level_config)` -> `{move_uci, san, eval_cp, mate}` │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            chess_game.py + chess_vision.py (Rendue Flèche AR 3D/2D)         │
│  - État `WAIT_BOT_EXECUTION` : stocke `pending_bot_move`                    │
│  - Dessine en perspective (via `H_inv`) :                                   │
│      • Case départ en Orange pulsé + Case arrivée en Vert émeraude          │
│      • Flèche épaisse avec pointe triangulaire projetée en perspective      │
│      • Si capture (`is_capture=True`), icône/cercle rouge sur la cible      │
│  - Vérifie automatiquement via `compute_square_changes` quand l'humain      │
│    a exécuté `pending_bot_move` sur le plateau physique !                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Plan d'Implémentation Détaillé (Fichier par Fichier)

### Tâche 1 : Créer `chess_engine.py` (`ChessEngineManager`)

Créer `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\chess_engine.py` :

1. **Détection de Stockfish (`_find_stockfish_binary()`)** :
   - Cherche dans l'ordre : variable d'environnement `STOCKFISH_PATH`, dossier `engines/stockfish*.exe` (Windows) ou `engines/stockfish*`, `shutil.which("stockfish")`, `/usr/games/stockfish` (emplacement standard Raspberry Pi / Debian via `sudo apt install stockfish`).
   - Permet aussi à l'utilisateur de renseigner le chemin vers `stockfish.exe` depuis l'UI Web.

2. **Moteur Python Intégré (`_builtin_alphabeta_search(board, depth, noise_cp)`)** :
   - Valeurs matérielles : Pion = `100`, Cavalier = `320`, Fou = `330`, Tour = `500`, Dame = `900`, Roi = `20000`.
   - Tables de bonus positionnels (Piece-Square Tables 8×8) encourageant le contrôle du centre (`d4, e4, d5, e5`), le développement des cavaliers/fous et la sécurité du roi roqué.
   - Tri des coups (MVV-LVA : captures en premier) pour accélérer l'élagage Alpha-Beta (tourne en < 150ms à profondeur 3 sur un Raspberry Pi !).
   - Injection d'imprécision humaine (`noise_cp`) : ajoute un bruit gaussien `random.gauss(0, noise_cp)` au score de chaque coup racine afin que les niveaux 800–1200 ELO jouent de vrais coups variés et fassent parfois des erreurs exploitables par le joueur.

3. **Méthode `compute_move(board_fen, elo_target=1200, time_limit=0.4)`** :
   - Si `elo_target <= 1300` ou que Stockfish n'est pas installé : utilise le moteur Alpha-Beta calibré (`depth=1..3`, `noise_cp` adapté à `elo_target`).
   - Si Stockfish est disponible et `elo_target > 1300` : configure `UCI_LimitStrength=True`, `UCI_Elo=min(max(elo_target, 1320), 3190)` (ou `Skill Level` de `0` à `20`) et retourne le coup + l'évaluation en centipions (`eval_cp`).

4. **Méthode `get_hint(board_fen)`** :
   - Calcule le meilleur coup possible pour le joueur actif (avec la profondeur maximale disponible) et retourne `{move_uci, san, eval_cp, explanation}`.

---

### Tâche 2 : Rendu de la Flèche en Perspective dans `chess_vision.py`

Ajouter dans `ChessVisionManager` la méthode `draw_perspective_arrow(display_frame, from_sq, to_sq, color_bgr, is_capture=False, label=None)` :
1. Dans le repère plat 512×512 :
   - Calculer le centre `(x1, y1)` de `from_sq` et le centre `(x2, y2)` de `to_sq`.
   - Construire les sommets géométriques 2D d'une belle flèche (tige de largeur `14px` + tête triangulaire de largeur `34px` et longueur `28px`) dans l'espace 512×512.
2. Transformer tous les sommets du polygone de la flèche ainsi que les 4 coins de `from_sq` et `to_sq` vers l'image caméra via `cv2.perspectiveTransform(pts, self.H_inv)` !
   - *Pourquoi construire la flèche dans l'espace 512×512 avant de la projeter avec `H_inv` ?* Parce qu'ainsi la flèche subit **exactement la même déformation de perspective 3D** que l'échiquier (plus large au premier plan, plus fine à l'arrière-plan), ce qui donne un effet de Réalité Augmentée "collé au plateau" bluffant !
3. Dessiner avec un calque semi-transparent (`cv2.fillPoly` + `cv2.addWeighted` avec `alpha = 0.55`) et un contour net anti-aliasé (`cv2.polylines(..., cv2.LINE_AA)`).
4. Si `is_capture=True` : dessiner une cible rouge autour de la case d'arrivée `to_sq` pour avertir le joueur (ou le robot) qu'il faut retirer la pièce adverse.
5. Si c'est un **roque (`e1g1`, `e1c1`, `e8g8`, `e8c8`)** : dessiner une flèche principale pour le Roi et une flèche secondaire en pointillés pour la Tour à déplacer !

---

### Tâche 3 : Intégrer le Cycle Humain $\leftrightarrow$ Bot dans `chess_game.py`

1. **Configuration de la partie** :
   - `self.game_mode = "HUMAN_VS_BOT"` (ou `"HUMAN_VS_HUMAN"` pour analyser une partie entre 2 amis dans le salon !).
   - `self.human_color = chess.WHITE` (ou `chess.BLACK` : si l'humain choisit les Noirs, le Bot joue dès le début de la partie).
   - `self.bot_elo = 1200` (réglable à chaud en cours de partie).
2. **Transition après un coup humain validé (`WAIT_HUMAN_MOVE` $\rightarrow$ `WAIT_BOT_THINKING` $\rightarrow$ `WAIT_BOT_EXECUTION`)** :
   - Dès que le coup de l'humain est détecté et poussé sur `self.board` :
     - Si `self.game_mode == "HUMAN_VS_BOT"` et que la partie n'est pas finie :
       - Lancer le calcul du Bot dans un thread court (`chess_engine.compute_move`) pour ne jamais bloquer le flux vidéo.
       - Stocker le résultat dans `self.pending_bot_move` (objet contenant `uci`, `san`, `from_sq`, `to_sq`, `is_capture`, `is_castling`).
       - Passer à l'état `"WAIT_BOT_EXECUTION"`.
3. **Validation automatique de l'exécution du coup du Bot (`check_bot_move_executed`)** :
   - Dans l'état `"WAIT_BOT_EXECUTION"`, dès que le plateau redevient stable après le passage de la main (ou du bras robot) :
   - Vérifier que les cases attendues de `self.pending_bot_move` ont changé au-dessus du seuil `MIN_CHANGE_THRESHOLD`.
   - Dès que c'est le cas :
     - Pousser `self.pending_bot_move` sur `self.board`.
     - Mettre à jour `self.reference_warped = cur_warped.copy()`.
     - Vider `self.pending_bot_move = None` et repasser à `"WAIT_HUMAN_MOVE"` !
   - Prévoir aussi un bouton **"✅ Forcer validation du coup Bot"** dans l'UI au cas où l'éclairage empêcherait la détection automatique.

---

### Tâche 4 : Routes API & Interface Web (Slice 3)

1. **Nouvelles Routes API dans `web_app.py`** :
   - `GET /api/engine/status` : indique si Stockfish est détecté ou si le moteur Python intégré est actif, le niveau ELO courant, le mode de jeu et la couleur du joueur.
   - `POST /api/engine/settings` : permet de modifier à chaud `bot_elo` (`600..2800`), `game_mode` (`"HUMAN_VS_BOT"` ou `"HUMAN_VS_HUMAN"`), `human_color` (`"white"` ou `"black"`), et optionnellement `stockfish_path`.
   - `POST /api/game/hint` : calcule un indice pour le joueur et l'affiche en AR (flèche bleue) pendant 8 secondes (ou jusqu'au prochain coup).
   - `POST /api/game/confirm-bot-move` : valide manuellement que le coup du bot a été déplacé sur le plateau et resynchronise `reference_warped`.
2. **Ajouts dans l'UI Web (`templates/index.html`, `style.css`, `app.js`)** :
   - **Sélecteur de Mode & Couleur** : `👤 Humain vs 🤖 Bot` / `👥 2 Joueurs + Coach AR`, choix `Blancs ♔` / `Noirs ♚`.
   - **Slider de Niveau ELO du Bot** : Curseur de `600 ELO (Débutant)` à `2600 ELO (Grand Maître)` avec badges prédéfinis (`Débutant 800`, `Club 1400`, `Avancé 1800`, `Max`).
   - **Bannière d'Action Courante très lisible** :
     - Ex: `"🟢 À votre tour de jouer (Blancs)"`
     - Ex: `"🟠 Déplacez la pièce du Bot : Cavalier g8 ➔ f6"`
   - **Jauge verticale d'évaluation** à gauche de l'échiquier 2D + bouton **`💡 Demander un Indice AR`**.

---

## 4. Vérification & Tests d'Acceptation (Slice 3)

1. **Test unitaire du Moteur (`chess_engine.py`)** :
   - Tester `compute_move()` sur la position initiale et sur une position de mat en 1 avec `elo_target=800` puis `elo_target=2000`.
   - Vérifier que le moteur répond en moins de `0.5s` sur CPU même sans Stockfish installé.
2. **Test du Rendu de la Flèche AR en Perspective** :
   - Simuler un `pending_bot_move = e7e5` et vérifier que `process_frame()` dessine bien la flèche projetée entre `e7` et `e5` sans erreur OpenCV.
3. **Test de Partie Complète** :
   - Jouer 3 coups complets (Humain $\rightarrow$ Flèche AR Bot $\rightarrow$ Déplacement manuel de la pièce du Bot $\rightarrow$ Validation automatique $\rightarrow$ Tour Humain).
