# Chess AR Vision

Jeu d'échecs en réalité augmentée, piloté par une webcam. Le projet reconnaît l'échiquier et les coups joués à la main, fait jouer un moteur d'échecs, et projette les indices et les coups du bot directement sur le plateau réel.

Le projet tourne en local, sans GPU, avec Python, Flask et OpenCV. Il est conçu pour fonctionner sur un PC portable comme sur un Raspberry Pi.

## Fonctionnalités

- **Calibration du plateau** : détection automatique des 4 coins par marqueurs ArUco, par transformée de Hough, ou par les deux combinés. Réglage manuel possible par glisser-déposer dans l'interface.
- **Vue redressée** : l'échiquier est ramené en vue de dessus (512×512) par homographie, avec la grille `a1..h8` projetée sur le plateau réel.
- **Suivi des coups** : détection d'un coup joué à la main par différence d'images, puis validation contre les coups légaux (`python-chess`).
- **Moteur de jeu** : bot avec niveau réglable. Il utilise Stockfish via UCI s'il est installé, sinon un moteur Python alpha-bêta intégré.
- **Réalité augmentée** : flèches et indices projetés sur le plateau, mode Coach.
- **Sauvegarde** : l'état de la partie est enregistré dans `game_state.json` à chaque coup ou modification manuelle.
- **Pont robotique** (en cours) : traduction d'un coup en séquence d'actions pour le bras SO-101 (`robot_bridge.py`).
- **Classification ONNX** (en cours) : classifieur de pièces entraînable sur GPU, exécuté sur CPU via `cv2.dnn`.

## Prérequis

- Python 3.10 ou plus récent
- Une webcam
- Un plateau d'échecs, idéalement avec les marqueurs ArUco de `static/aruco/` (planche imprimable : `planche_aruco_imprimable.png`)
- Optionnel : [Stockfish](https://stockfishchess.org/) pour un bot plus fort

## Installation

```bash
git clone https://github.com/BenjiKo14/Chess_AR_Vision.git
cd Chess_AR_Vision
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Lancement

Sous Windows, double-cliquer sur `LANCER_SERVEUR.bat`, ou exécuter :

```bash
python start_web.py
```

Le serveur écoute sur `http://127.0.0.1:5001`. Le navigateur s'ouvre automatiquement.

Le port 5001 permet de faire tourner le projet en parallèle de `Robot_Arm_SO-101`, qui utilise le port 5000.

## Utilisation

1. Placer la planche ArUco ou l'échiquier dans le champ de la caméra.
2. Lancer la calibration. Vérifier que la vue redressée et la grille AR correspondent au plateau réel. Corriger les coins à la main si besoin.
3. Installer les pièces en position de départ, puis jouer un coup à la main. Le coup est détecté et validé.
4. Le bot répond, et son coup s'affiche en réalité augmentée.

## Structure du projet

| Fichier | Rôle |
| :--- | :--- |
| `start_web.py` | Point d'entrée : lance le serveur et ouvre le navigateur |
| `web_app.py` | Serveur Flask, API REST, flux MJPEG |
| `camera.py` | Gestion de la webcam dans un thread dédié |
| `chess_vision.py` | Calibration, homographie, différence entre cases, overlays AR |
| `hough_board.py` | Détection du damier par transformée de Hough |
| `board_reader.py` | Lecture de la position sur l'image redressée |
| `chess_game.py` | Machine à états de la partie, validation des coups |
| `chess_engine.py` | Moteur de jeu (Stockfish ou moteur Python intégré) |
| `robot_bridge.py` | Décomposition des coups en actions robotiques |
| `dataset_collector.py` | Collecte et labellisation automatique des cases |
| `onnx_classifier.py` | Classifieur de pièces ONNX (CPU) |
| `train_piece_classifier.py` | Entraînement du classifieur (GPU) |
| `tools_grid_eval.py` | Outil d'évaluation de l'alignement du quadrillage |
| `board_calibration.json` | Coins du plateau et orientation sauvegardés |
| `specs/` | Spécifications détaillées de chaque étape du projet |
| `ROADMAP.md` | Découpage du projet en 5 slices |

## Tests

```bash
python test_integration.py
python test_slice5.py
```

## Feuille de route

Le projet est découpé en cinq slices, détaillées dans `specs/` et `ROADMAP.md` :

1. Socle web, caméra et calibration du plateau
2. Détection des coups joués à la main
3. Moteur de jeu et gameplay en réalité augmentée
4. Pont avec le bras robotique SO-101
5. Fine-tuning MobileNetV3 et inférence ONNX sur CPU

## Licence

Aucune licence n'est définie pour le moment.
