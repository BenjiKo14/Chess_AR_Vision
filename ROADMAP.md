# Chess_AR_Vision — Roadmap & Index des Spécifications par Slices

**Projet** : `C:\Users\b14ko\Documents\Projets\Chess_AR_Vision`  
**Projet frère (fusion Phase 2)** : `C:\Users\b14ko\Documents\Projets\Robot_Arm_SO-101`  
**Stack Cible** : Python 3 + Flask + OpenCV (`cv2`) + NumPy + `python-chess` (100% CPU, compatible PC portable & Raspberry Pi, zéro GPU requis)  
**Port Serveur Web** : `http://127.0.0.1:5001` (pour pouvoir tourner en parallèle de `Robot_Arm_SO-101` qui utilise le port `5000`)

---

## Vue d'ensemble du Découpage en 5 Slices

Chaque Slice est autonome, testable de bout en bout, et documentée dans un fichier de spécification ultra-détaillé pour permettre une implémentation directe par un modèle léger.

| Slice | Fichier de Spécification | Objectif Principal | Livrables Clés |
| :--- | :--- | :--- | :--- |
| **Slice 1** | [`specs/SPEC_SLICE1_SOCLE_ET_CALIBRATION.md`](file:///C:/Users/b14ko/Documents/Projets/Chess_AR_Vision/specs/SPEC_SLICE1_SOCLE_ET_CALIBRATION.md) | **Socle Web, Caméra & Calibration Perspective ($H / H^{-1}$)** | `camera.py`, `chess_vision.py` (calibration auto + 4 coins manuels drag&drop, warp 512×512, grille AR projetée), `web_app.py`, UI Web |
| **Slice 2** | [`specs/SPEC_SLICE2_SUIVI_COUPS_VISION.md`](file:///C:/Users/b14ko/Documents/Projets/Chess_AR_Vision/specs/SPEC_SLICE2_SUIVI_COUPS_VISION.md) | **Détection de Mouvement & Suivi des Coups par Différence (Approche B)** | Détecteur de main/mouvement, guide AR de placement initial, extraction des 64 cases, matching avec les coups légaux `python-chess` (`chess_game.py`) |
| **Slice 3** | [`specs/SPEC_SLICE3_MOTEUR_ET_AR_GAMEPLAY.md`](file:///C:/Users/b14ko/Documents/Projets/Chess_AR_Vision/specs/SPEC_SLICE3_MOTEUR_ET_AR_GAMEPLAY.md) | **Moteur Multi-Niveaux (Stockfish + Fallback) & Gameplay AR Complet** | `chess_engine.py` (réglage ELO/Skill), flèche AR projetée en perspective, détection auto du déplacement de la pièce du bot par l'humain, mode Coach/Indice |
| **Slice 4** | [`specs/SPEC_SLICE4_CONTRAT_ROBOT_SO101.md`](file:///C:/Users/b14ko/Documents/Projets/Chess_AR_Vision/specs/SPEC_SLICE4_CONTRAT_ROBOT_SO101.md) | **Pont Robotique & Fusion avec `Robot_Arm_SO-101`** | `robot_bridge.py`, décomposition d'un coup d'échecs en primitives robotiques (prise, capture vers cimetière, roque), coordonnées normalisées & API d'intégration |
| **Slice 5** | [`specs/SPEC_SLICE5_FINETUNING_MOBILENET_ONNX.md`](file:///C:/Users/b14ko/Documents/Projets/Chess_AR_Vision/specs/SPEC_SLICE5_FINETUNING_MOBILENET_ONNX.md) | **Fine-Tuning GPU (`MobileNetV3`) & Scan Milieu de Partie CPU (`ONNX`)** | `dataset_collector.py` (auto-labellisation des 64 cases), `train_piece_classifier.py` (GPU $\rightarrow$ `.onnx`), `onnx_classifier.py` (`cv2.dnn` 100% CPU en < 50ms) |

---

## Arborescence Finale Cible du Projet

```text
C:\Users\b14ko\Documents\Projets\Chess_AR_Vision\
├── ROADMAP.md
├── requirements.txt
├── start_web.py
├── LANCER_SERVEUR.bat
├── web_app.py                  # Serveur Flask + API REST + Flux MJPEG (caméra AR + vue 2D redressée)
├── camera.py                   # Gestionnaire webcam threadé (compatible Robot_Arm_SO-101)
├── chess_vision.py             # Calibration Auto/Manuelle, Homographie H et H_inv, Diff 64 cases, Overlays AR
├── chess_game.py               # Machine à états de la partie + python-chess + validation des coups physiques
├── chess_engine.py             # Bot modulable (Stockfish UCI avec ELO/Skill + moteur Python alpha-beta intégré)
├── robot_bridge.py             # Traducteur de coups d'échecs en séquences d'actions pour Robot_Arm_SO-101
├── board_calibration.json      # Persistance des 4 coins [A8, H8, H1, A1] et orientation
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
└── specs/
    ├── SPEC_SLICE1_SOCLE_ET_CALIBRATION.md
    ├── SPEC_SLICE2_SUIVI_COUPS_VISION.md
    ├── SPEC_SLICE3_MOTEUR_ET_AR_GAMEPLAY.md
    └── SPEC_SLICE4_CONTRAT_ROBOT_SO101.md
```

---

## Ordre d'Exécution Recommandé

1. Demander au modèle d'exécuter **`specs/SPEC_SLICE1_SOCLE_ET_CALIBRATION.md`** et valider dans le navigateur que la caméra s'affiche, que les 4 coins (auto ou drag & drop) redressent bien l'échiquier dans la miniature 512×512 et projettent la grille AR `a1..h8` sur le vrai plateau.
2. Exécuter **`specs/SPEC_SLICE2_SUIVI_COUPS_VISION.md`** et tester la détection d'un coup réel joué à la main sur le plateau.
3. Exécuter **`specs/SPEC_SLICE3_MOTEUR_ET_AR_GAMEPLAY.md`** pour jouer une partie complète contre le bot avec les flèches en Réalité Augmentée.
4. Exécuter **`specs/SPEC_SLICE4_CONTRAT_ROBOT_SO101.md`** lorsque tu es prêt à connecter le bras robotique.
