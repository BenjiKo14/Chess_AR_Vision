import os
import sys
import json
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import chess
from pastille_detector import detect_pastilles_by_square, detect_pastille_blobs, PASTILLE_PAD

import hough_board

def safe_log_print(msg):
    """Affiche un message en console de façon sûre (évite UnicodeEncodeError sous Windows cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
            print(str(msg).encode(enc, errors='replace').decode(enc))
        except Exception:
            try:
                print(str(msg).encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
    except Exception:
        pass

WARP_SIZE = 512
SQUARE_SIZE = WARP_SIZE // 8  # 64 px

DEFAULT_DETECTION_CONFIG = {
    "min_side_px": 8,
    "max_side_px": 52,
    "clahe_clip": 3.5,
    "bin_thresh": 125,
    "border_err": 0.85,
    "err_corr": 0.85,
    "aruco_inset_pct": 2.4,  # Retrait automatique (%) des marqueurs vers les coins du damier 8x8
    "use_hough_refine": True, # Couplage avec la transformée de Hough pour calage subpixel sur les lignes de cases
    "detection_method": "COUPLED", # 'ARUCO', 'HOUGH', 'COUPLED'
    "grid_max_shift": 8,
    "piece_mode": "PASTILLES_STRICT", # 'PASTILLES_STRICT' (recommandé), 'HYBRID'
    "pastille_min_area": 45,
    "pastille_max_area": 320,
    "pastille_min_circ": 0.52,
    "pastille_min_sat": 40,
    "parallax_scale": 1.0,
}

DEFAULT_SETTINGS = {
    "show_ar_grid": True,
    "show_ar_labels": True,
    "show_ar_arrow": True,
    "show_ar_vision_debug": False,
    "show_ar_pieces": False,
    "show_warped_grid": True,
    "show_warped_arrow": True,
    "show_aruco_debug": True,
}

class ChessVisionManager:
    """
    Gère la calibration géométrique, l'homographie perspective H et H_inv,
    le découpage des 64 cases, l'analyse différentielle de stabilité et les overlays AR.
    """
    def __init__(self, calibration_file="board_calibration.json", log_fn=None):
        self.calibration_file = Path(calibration_file)
        self.log_fn = log_fn or safe_log_print
        self._lock = threading.RLock()

        # 4 coins normalisés [0.0 .. 1.0] dans l'ordre strict horaire :
        # 0: A8 (Haut-Gauche), 1: H8 (Haut-Droite), 2: H1 (Bas-Droite), 3: A1 (Bas-Gauche)
        self.corners_norm = [
            [0.20, 0.15],
            [0.80, 0.15],
            [0.85, 0.85],
            [0.15, 0.85]
        ]
        self.is_calibrated = False
        # Incrémenté à chaque changement de géométrie ou d'image (coins, quadrillage, caméra) :
        # une image de référence prise avant n'est plus comparable case par case.
        self.geometry_version = 0

        # Cache homographie
        self.last_frame_shape = None
        self.H = None
        self.H_inv = None

        # Images mémorisées
        self._latest_warped_frame = None      # Image NumPy BGR 512x512
        self._latest_warped_jpeg = None       # Bytes JPEG 512x512
        self._reference_warped_frame = None   # Image 512x512 référence début de tour
        self._prev_warped_gray = None

        # Options d'affichage AR
        self.show_ar_grid = True
        self.show_ar_labels = True
        self.show_ar_arrow = True      # Flèche AR sur l'écran caméra principal
        self.show_ar_vision_debug = False # Vue détection ordinateur (carte des contours/gradients projetée)
        self.show_ar_pieces = False       # Pastilles des pièces détectées (Blancs/Noirs) en direct
        self.show_warped_grid = True
        self.show_warped_arrow = True  # Flèche sur la vue plateau redressée 512x512
        self.show_aruco_debug = True  # Affiche les carrés ArUco détectés avec leur contour vert et leur ID
        self.last_detected_markers = {}  # Stocke id -> np.array (4x2)
        self.last_aruco_centers_norm = None
        self.last_detected_pastilles = {}   # sq -> PastilleDetection
        self.continuous_detection = False
        self.last_continuous_detection_time = 0.0  # Stocke les 4 centres bruts des marqueurs pour ajuster l'inset en direct
        self.last_detected_pieces = None    # Stocke les probabilités de présence par case [P_vide, P_blanc, P_noir]
        self.current_game_board = None      # Instance de chess.Board du jeu en cours (référence de vérité)
        self._ar_piece_tracker = {}         # sq -> dict de suivi AR avec debounce, ttl et lissage sub-pixel
        self._ar_frame_count = 0            # Compteur de frames traitées
        # Lignes directrices exactes détectées par Hough (9 lignes horizontales et 9 lignes verticales dans 512x512)
        # Quadrillage interne affiné (lignes internes calées sur les vraies séparations ; contour jamais modifié)
        self.refined_grid = None
        self.refined_grid_count = 0
        self.hough_grid_lines = {
            'rows': [float(i * SQUARE_SIZE) for i in range(9)],
            'cols': [float(j * SQUARE_SIZE) for j in range(9)]
        }

        # Console de configuration des paramètres de détection (réglables depuis l'UI Web, avec persistance)
        self.detection_config = DEFAULT_DETECTION_CONFIG.copy()

        # Stabilité et mouvement (MotionGate)
        self.is_board_stable = True
        self.motion_ratio = 0.0
        self.last_motion_time = time.time()
        self.stability_delay = 0.75

        # Support ArUco multi-dictionnaires
        self.aruco_dicts = [
            ("DICT_4X4_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)),
            ("DICT_4X4_100", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
            ("DICT_5X5_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)),
            ("DICT_6X6_50", cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)),
        ]
        self.aruco_detectors = []
        if hasattr(cv2.aruco, 'ArucoDetector'):
            params = cv2.aruco.DetectorParameters()
            # Optimisations pour éclairage de salon et petits marqueurs
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 35
            params.adaptiveThreshWinSizeStep = 4
            params.minMarkerPerimeterRate = 0.015 # Permet de détecter les petits marqueurs même loin
            params.maxMarkerPerimeterRate = 4.0
            params.polygonalApproxAccuracyRate = 0.05
            for name, d in self.aruco_dicts:
                self.aruco_detectors.append((name, cv2.aruco.ArucoDetector(d, params)))
        else:
            self.aruco_detectors = []
        self.auto_aruco_tracking = True

        # Éléments AR dynamiques fournis par ChessGameManager
        self.ar_arrow = None             # Dict: {"from_sq": int, "to_sq": int, "is_capture": bool, "color": tuple, "label": str}
        self.highlight_squares = {}      # sq -> {"color": (B, G, R), "alpha": float}
        self.setup_mode = False          # Calque guide départ
        self.top_diff_info = ""

        # Chargement de la calibration et des configurations personnalisées
        self._load_calibration()

    def log(self, msg):
        try:
            if self.log_fn:
                self.log_fn(msg)
            else:
                safe_log_print(msg)
        except UnicodeEncodeError:
            safe_log_print(msg)
        except Exception:
            safe_log_print(msg)

    def _load_calibration(self):
        try:
            if self.calibration_file.exists():
                with open(self.calibration_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'corners' in data and len(data['corners']) == 4:
                        self.corners_norm = data['corners']
                        self.is_calibrated = True
                    # Restauration de la configuration de détection personnalisée
                    if 'detection_config' in data and isinstance(data['detection_config'], dict):
                        for k, v in data['detection_config'].items():
                            if k in self.detection_config:
                                self.detection_config[k] = v
                    # Restauration des options d'overlay utilisateur
                    if 'settings' in data and isinstance(data['settings'], dict):
                        s = data['settings']
                        for k in ('show_ar_grid', 'show_ar_labels', 'show_ar_arrow', 'show_ar_vision_debug',
                                  'show_ar_pieces', 'show_warped_grid', 'show_warped_arrow', 'show_aruco_debug'):
                            if k in s:
                                setattr(self, k, bool(s[k]))
                    # Restauration de l'état de détection continue
                    if 'continuous_detection' in data:
                        self.continuous_detection = bool(data['continuous_detection'])
                    self.log("✅ Configuration et calibration plateau chargées depuis board_calibration.json")
        except Exception as e:
            self.log(f"⚠️ Erreur chargement calibration: {e}")

    def save_calibration(self):
        try:
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'corners': self.corners_norm,
                    'is_calibrated': self.is_calibrated,
                    'detection_config': self.detection_config,
                    'settings': {
                        'show_ar_grid': self.show_ar_grid,
                        'show_ar_labels': self.show_ar_labels,
                        'show_ar_arrow': self.show_ar_arrow,
                        'show_ar_vision_debug': self.show_ar_vision_debug,
                        'show_ar_pieces': self.show_ar_pieces,
                        'show_warped_grid': self.show_warped_grid,
                        'show_warped_arrow': self.show_warped_arrow,
                        'show_aruco_debug': self.show_aruco_debug
                    },
                    'continuous_detection': self.continuous_detection,
                    'updated_at': time.time()
                }, f, indent=2)
            self.log("💾 Configuration et calibration sauvegardées.")
        except Exception as e:
            self.log(f"❌ Erreur sauvegarde calibration: {e}")

    def set_corners(self, corners, force=False):
        """Met à jour les 4 coins normalisés [[x0,y0], ... [x3,y3]]."""
        with self._lock:
            if len(corners) == 4:
                new_corners = [[float(p[0]), float(p[1])] for p in corners]
                # Si déjà calibré, vérifier si le déplacement est significatif (> ~5-6 pixels)
                if not force and self.is_calibrated and self.corners_norm and len(self.corners_norm) == 4:
                    shifts = [
                        np.hypot(n[0] - o[0], n[1] - o[1])
                        for n, o in zip(new_corners, self.corners_norm)
                    ]
                    max_shift = max(shifts)
                    # Déplacement infime (bruit de caméra) : on ne perturbe pas la partie ni la référence
                    if max_shift < 0.008:
                        # Lissage exponentiel doux sans toucher à geometry_version ni au disque
                        self.corners_norm = [
                            [0.95 * o[0] + 0.05 * n[0], 0.95 * o[1] + 0.05 * n[1]]
                            for n, o in zip(new_corners, self.corners_norm)
                        ]
                        return False

                self.corners_norm = new_corners
                self.is_calibrated = True
                self.last_frame_shape = None
                self.refined_grid = None  # le quadrillage affiné dépend du contour : à recalculer
                self.geometry_version += 1
                self.save_calibration()
                return True
        return False

    def rotate_board_90(self):
        """Pivote circulairement les 4 coins pour réaligner les Blancs."""
        with self._lock:
            # Rotation horaire: A8 devient H8 etc.
            self.corners_norm = [self.corners_norm[1], self.corners_norm[2], self.corners_norm[3], self.corners_norm[0]]
            self.last_frame_shape = None
            self.refined_grid = None
            self.geometry_version += 1
            self.save_calibration()
            return self.corners_norm

    def update_homography(self, shape):
        """Met à jour les matrices d'homographie H et H_inv pour une taille d'image (h, w)."""
        if len(shape) == 2:
            h, w = (shape[0], shape[1]) if shape[0] < shape[1] else (shape[1], shape[0])
            with self._lock:
                self._update_homography(w, h)
                return self.H is not None
        return False

    def warp_frame(self, frame):
        """Redresse une image brute selon l'homographie courante vers le repère 512x512."""
        h, w = frame.shape[:2]
        with self._lock:
            self._update_homography(w, h)
            if self.H is not None:
                return cv2.warpPerspective(frame, self.H, (WARP_SIZE, WARP_SIZE))
            return None

    def _update_homography(self, w, h):
        shape = (w, h)
        if shape == self.last_frame_shape and self.H is not None:
            return

        src_pts = np.float32([[p[0] * w, p[1] * h] for p in self.corners_norm])
        dst_pts = np.float32([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]])

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        dst_pad = np.float32([
            [PASTILLE_PAD, PASTILLE_PAD],
            [WARP_SIZE + PASTILLE_PAD, PASTILLE_PAD],
            [WARP_SIZE + PASTILLE_PAD, WARP_SIZE + PASTILLE_PAD],
            [PASTILLE_PAD, WARP_SIZE + PASTILLE_PAD]
        ])
        self.H_pad = cv2.getPerspectiveTransform(src_pts, dst_pad)
        self.last_frame_shape = shape

    def warped_points_to_camera(self, pts_2d):
        """Projette une liste de points [(x, y), ...] du repère 512x512 vers l'image caméra."""
        if self.H_inv is None:
            return pts_2d
        pts = np.array(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return [(float(pt[0][0]), float(pt[0][1])) for pt in transformed]

    # --- Modèle de quadrillage (repère redressé 512x512) ---
    # Chaque ligne verticale k est un segment (x_haut, x_bas) ; chaque ligne horizontale k est (y_gauche, y_droite).
    # Les lignes 0 et 8 (contour) restent TOUJOURS fixées sur le contour détecté par ArUco.

    @staticmethod
    def _uniform_grid():
        return {
            'v': [(float(k * SQUARE_SIZE), float(k * SQUARE_SIZE)) for k in range(9)],
            'h': [(float(k * SQUARE_SIZE), float(k * SQUARE_SIZE)) for k in range(9)],
        }

    def _active_grid(self):
        """Retourne la grille à utiliser : grille affinée (Hough/gradient) si activée, sinon uniforme."""
        if self.detection_config.get("use_hough_refine", True) and getattr(self, 'refined_grid', None):
            return self.refined_grid
        return self._uniform_grid()

    def grid_point(self, k_col, k_row, grid=None):
        """Intersection de la ligne verticale k_col et de la ligne horizontale k_row (repère 512x512)."""
        g = grid or self._active_grid()
        xt, xb = g['v'][k_col]
        yl, yr = g['h'][k_row]
        # x = xt + (xb - xt) * y / S ; y = yl + (yr - yl) * x / S  -> résolution par itérations (converge très vite)
        S = float(WARP_SIZE)
        y = (yl + yr) / 2.0
        x = xt + (xb - xt) * y / S
        for _ in range(3):
            y = yl + (yr - yl) * x / S
            x = xt + (xb - xt) * y / S
        return float(x), float(y)

    def square_warped_polygon(self, col, row):
        """4 sommets (repère 512x512) de la case col 0..7 (a..h), row 0..7 (8..1)."""
        g = self._active_grid()
        return [self.grid_point(col, row, g), self.grid_point(col + 1, row, g),
                self.grid_point(col + 1, row + 1, g), self.grid_point(col, row + 1, g)]

    def square_to_warped_rect(self, col, row):
        """
        Retourne (x1, y1, x2, y2) dans l'image 512x512 pour col in 0..7 (a..h) et row in 0..7 (8..1).
        Basé sur la grille active (les cases peuvent différer de quelques pixels).
        """
        poly = self.square_warped_polygon(col, row)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1 = int(round((xs[0] + xs[3]) / 2.0))
        x2 = int(round((xs[1] + xs[2]) / 2.0))
        y1 = int(round((ys[0] + ys[1]) / 2.0))
        y2 = int(round((ys[2] + ys[3]) / 2.0))
        x1, x2 = max(0, min(x1, WARP_SIZE)), max(0, min(x2, WARP_SIZE))
        y1, y2 = max(0, min(y1, WARP_SIZE)), max(0, min(y2, WARP_SIZE))
        return x1, y1, x2, y2

    def square_center_warped(self, col, row):
        poly = self.square_warped_polygon(col, row)
        return (sum(p[0] for p in poly) / 4.0, sum(p[1] for p in poly) / 4.0)

    def square_to_camera_polygon(self, col, row):
        """Retourne les 4 sommets projetés d'une case sur l'image caméra."""
        pts = self.square_warped_polygon(col, row)
        return np.array(self.warped_points_to_camera(pts), dtype=np.int32)

    def _apply_inset_to_centers(self, centers_norm, inset_pct=None):
        """
        Convertit les 4 centres des marqueurs ArUco (posés sur le cadre en bois)
        en les 4 coins exacts du damier 8x8 grâce au retrait projectif `inset_pct` (%).
        """
        if inset_pct is None:
            inset_pct = float(self.detection_config.get("aruco_inset_pct", 2.4))
        if abs(inset_pct) < 1e-4:
            return [[float(p[0]), float(p[1])] for p in centers_norm]

        src = np.float32(centers_norm)
        d = (inset_pct / 100.0) * float(WARP_SIZE)
        dst_markers = np.float32([
            [-d, -d],
            [WARP_SIZE + d, -d],
            [WARP_SIZE + d, WARP_SIZE + d],
            [-d, WARP_SIZE + d]
        ])
        H_m2b = cv2.getPerspectiveTransform(dst_markers, src)
        board_dst = np.float32([[[0, 0]], [[WARP_SIZE, 0]], [[WARP_SIZE, WARP_SIZE]], [[0, WARP_SIZE]]])
        board_norm = cv2.perspectiveTransform(board_dst, H_m2b).reshape(4, 2)
        return [[float(p[0]), float(p[1])] for p in board_norm]

    @staticmethod
    def _robust_line_fit(samples, nominal, max_shift, center=WARP_SIZE / 2.0):
        """
        Ajuste pos = a + b*(t - center) sur des échantillons [(t, pos, poids)] avec rejet des points
        aberrants (pièces, reflets). Retourne (pos_début, pos_fin) aux extrémités t=0 et t=WARP_SIZE.
        """
        if len(samples) < 3:
            return None
        pts = np.array(samples, dtype=np.float64)
        t, p, wgt = pts[:, 0] - center, pts[:, 1], pts[:, 2]
        best_inl = None
        # RANSAC exhaustif sur les paires (au plus 8 échantillons -> 28 paires)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(t[j] - t[i]) < 1e-6:
                    continue
                b = (p[j] - p[i]) / (t[j] - t[i])
                a = p[i] - b * t[i]
                inl = np.abs(p - (a + b * t)) < 1.8
                if best_inl is None or inl.sum() > best_inl.sum() or \
                   (inl.sum() == best_inl.sum() and wgt[inl].sum() > wgt[best_inl].sum()):
                    best_inl = inl
        if best_inl is None or best_inl.sum() < 4:
            # Pas assez de points alignés : simple décalage si 3 points cohérents
            med = float(np.median(p))
            close = np.abs(p - med) < 1.8
            if close.sum() >= 3:
                off = float(np.average(p[close], weights=wgt[close]))
                off = float(np.clip(off, nominal - max_shift, nominal + max_shift))
                return off, off
            return None
        A = np.stack([np.ones(best_inl.sum()), t[best_inl]], axis=1)
        W = np.sqrt(wgt[best_inl])
        sol, *_ = np.linalg.lstsq(A * W[:, None], p[best_inl] * W, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        p0 = float(np.clip(a + b * (0 - center), nominal - max_shift, nominal + max_shift))
        p1 = float(np.clip(a + b * (WARP_SIZE - center), nominal - max_shift, nominal + max_shift))
        return p0, p1

    def fit_grid_lines(self, frame, corners_norm=None):
        """
        Aligne les 7 lignes verticales et 7 lignes horizontales INTERNES du damier sur les vraies
        séparations de cases, sans jamais toucher au contour (coins ArUco).

        Méthode (robuste aux pièces) : dans la vue redressée, pour chaque ligne et pour chacun des
        8 segments qui la composent, on cherche le pic du gradient (Sobel) perpendiculaire dans une
        fenêtre ±max_shift px autour de la position théorique. Une droite (inclinaison autorisée)
        est ensuite ajustée par RANSAC sur les 8 pics, ce qui élimine les segments masqués par une pièce.
        """
        if frame is None:
            return False
        try:
            corners = corners_norm or self.corners_norm
            h, w = frame.shape[:2]
            src_pts = np.float32([[p[0] * w, p[1] * h] for p in corners])
            dst_pts = np.float32([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]])
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE), flags=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

            T = int(self.detection_config.get("grid_max_shift", 8))
            margin = 10  # on ignore les abords des intersections
            grid = self._uniform_grid()
            n_fitted = 0

            for k in range(1, 8):
                nominal = float(k * SQUARE_SIZE)
                lo, hi = int(nominal) - T, int(nominal) + T + 1
                for axis in ('v', 'h'):
                    samples = []
                    for s in range(8):
                        a0, a1 = s * SQUARE_SIZE + margin, (s + 1) * SQUARE_SIZE - margin
                        if axis == 'v':
                            prof = gx[a0:a1, lo:hi].mean(axis=0)
                        else:
                            prof = gy[lo:hi, a0:a1].mean(axis=1)
                        i = int(np.argmax(prof))
                        med = float(np.median(prof)) + 1e-3
                        strength = float(prof[i]) / med
                        if strength < 1.6 or i == 0 or i == len(prof) - 1:
                            continue
                        # Affinage sous-pixel parabolique
                        y0, y1, y2 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
                        den = (y0 - 2 * y1 + y2)
                        sub = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
                        pos = lo + i + float(np.clip(sub, -0.5, 0.5))
                        samples.append(((a0 + a1) / 2.0, pos, strength))
                    fit = self._robust_line_fit(samples, nominal, T)
                    if fit is not None:
                        grid[axis][k] = fit
                        n_fitted += 1

            with self._lock:
                self.refined_grid = grid
                self.geometry_version += 1
                self.refined_grid_count = n_fitted
            self.log(f"📐 Quadrillage aligné : {n_fitted}/14 lignes internes calées (contour inchangé)")
            return n_fitted > 0
        except Exception as e:
            self.log(f"Alignement quadrillage impossible : {e}")
            return False

    def invalidate_geometry(self):
        """À appeler quand l'image change de nature (caméra reconnectée, rotation, filtres)."""
        with self._lock:
            self.geometry_version += 1

    def reset_grid(self):
        with self._lock:
            self.refined_grid = None
            self.refined_grid_count = 0
            self.geometry_version += 1

    def reset_detection_config_to_defaults(self, raw_frame=None):
        """Réinitialise la console de détection aux valeurs par défaut d'usine, sauvegarde et applique."""
        with self._lock:
            self.detection_config = DEFAULT_DETECTION_CONFIG.copy()
            if self.last_aruco_centers_norm and len(self.last_aruco_centers_norm) == 4:
                self.set_corners(self._apply_inset_to_centers(self.last_aruco_centers_norm))
            self.save_calibration()

        if raw_frame is not None:
            self.auto_detect_board(raw_frame)
        return self.detection_config

    def update_detection_config(self, new_cfg, raw_frame=None):
        """Met à jour les paramètres de la console de détection, les persiste sur disque et recalcule si nécessaire."""
        if new_cfg.get("reset_to_default"):
            return self.reset_detection_config_to_defaults(raw_frame=raw_frame)

        with self._lock:
            for k in ("min_side_px", "max_side_px", "clahe_clip", "bin_thresh", "border_err", "err_corr", "aruco_inset_pct", "grid_max_shift"):
                if k in new_cfg and new_cfg[k] is not None:
                    self.detection_config[k] = float(new_cfg[k])
            if "detection_method" in new_cfg and new_cfg["detection_method"]:
                method_val = str(new_cfg["detection_method"]).upper()
                self.detection_config["detection_method"] = method_val
                # En mode ARUCO pur, on désactive le raffinement Hough pour garder la grille 8x8 uniforme
                # En mode COUPLED ou HOUGH, le raffinement des lignes Hough internes est actif
                self.detection_config["use_hough_refine"] = (method_val != "ARUCO")
            elif "use_hough_refine" in new_cfg:
                self.detection_config["use_hough_refine"] = bool(new_cfg["use_hough_refine"])

            # L'inset déplace le contour : on le recalcule depuis les centres ArUco mémorisés
            if "aruco_inset_pct" in new_cfg and self.last_aruco_centers_norm and len(self.last_aruco_centers_norm) == 4:
                self.set_corners(self._apply_inset_to_centers(self.last_aruco_centers_norm))

            # Sauvegarde persistante de la nouvelle configuration
            self.save_calibration()

        if raw_frame is not None and any(k in new_cfg for k in ("min_side_px", "max_side_px", "clahe_clip", "bin_thresh", "border_err", "err_corr", "detection_method")):
            self.auto_detect_board(raw_frame)
        elif raw_frame is not None and self.detection_config.get("use_hough_refine", True):
            # Le quadrillage interne est recalculé sur le contour courant
            self.fit_grid_lines(raw_frame)
        elif not self.detection_config.get("use_hough_refine", True):
            self.reset_grid()

        return self.detection_config

    def detect_aruco_corners(self, frame):
        """
        Détecte les 4 marqueurs aux coins en temps réel (< 50 ms) :
        - Détection directe sur image native (Gris + CLAHE léger)
        - Complétion géométrique projective instantanée si 3 marqueurs sur 4 sont vus (ex: pièce posée sur un marqueur)
        - Vérification de cohérence géométrique (convexité et ratios)
        """
        if frame is None:
            return False, None, "Pas d'image reçue"

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        cfg = self.detection_config
        min_side = float(cfg.get("min_side_px", 8))
        max_side = float(cfg.get("max_side_px", 55))
        clahe_clip = float(cfg.get("clahe_clip", 3.0))

        found_markers = {}
        marker_polys = {}

        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.012
        params.maxMarkerPerimeterRate = 0.30
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 25
        params.adaptiveThreshWinSizeStep = 4
        params.maxErroneousBitsInBorderRate = 0.55
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        det = cv2.aruco.ArucoDetector(d, params)

        # 1. Détection rapide à l'échelle native (1.0) sur Gray puis CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(gray)
        for src in (gray, clahe):
            corners, ids, _ = det.detectMarkers(src)
            if ids is not None:
                for m, c in zip(ids.flatten(), corners):
                    mid = int(m)
                    if mid in (0, 1, 2, 3) and mid not in found_markers:
                        pts = c[0]
                        side = float(np.linalg.norm(pts[0] - pts[1]))
                        if min_side <= side <= max_side:
                            center = pts.mean(axis=0)
                            found_markers[mid] = (float(center[0]), float(center[1]))
                            marker_polys[mid] = pts
            if len(found_markers) == 4:
                break

        # 2. Complétion géométrique projective si 3 marqueurs sur 4 sont détectés (< 1 ms)
        # Très fréquent si une pièce ou une main empiète temporairement sur un seul marqueur
        if len(found_markers) == 3:
            p = {k: np.array(found_markers[k], dtype=np.float32) for k in found_markers}
            if 0 in p and 1 in p and 3 in p and 2 not in p:
                # 2 (Bas-Droite) = 1 + (3 - 0)
                inferred = p[1] + (p[3] - p[0])
                found_markers[2] = (float(inferred[0]), float(inferred[1]))
            elif 0 in p and 1 in p and 2 in p and 3 not in p:
                # 3 (Bas-Gauche) = 0 + (2 - 1)
                inferred = p[0] + (p[2] - p[1])
                found_markers[3] = (float(inferred[0]), float(inferred[1]))
            elif 1 in p and 2 in p and 3 in p and 0 not in p:
                # 0 (Haut-Gauche) = 1 + (3 - 2)
                inferred = p[1] + (p[3] - p[2])
                found_markers[0] = (float(inferred[0]), float(inferred[1]))
            elif 0 in p and 2 in p and 3 in p and 1 not in p:
                # 1 (Haut-Droite) = 0 + (2 - 3)
                inferred = p[0] + (p[2] - p[3])
                found_markers[1] = (float(inferred[0]), float(inferred[1]))

        # 3. Passe légère de secours x1.25 seulement si < 3 marqueurs
        if len(found_markers) < 3:
            up = cv2.resize(gray, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LINEAR)
            up_clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(up)
            corners, ids, _ = det.detectMarkers(up_clahe)
            if ids is not None:
                for m, c in zip(ids.flatten(), corners):
                    mid = int(m)
                    if mid in (0, 1, 2, 3) and mid not in found_markers:
                        pts = c[0] / 1.25
                        center = pts.mean(axis=0)
                        found_markers[mid] = (float(center[0]), float(center[1]))
                        marker_polys[mid] = pts

        self.last_detected_markers = marker_polys
        detected_ids = sorted(list(found_markers.keys()))

        if len(found_markers) == 4 and all(k in found_markers for k in (0, 1, 2, 3)):
            # 0: Haut-Gauche, 1: Haut-Droite, 2: Bas-Droite, 3: Bas-Gauche
            tl = np.array(found_markers[0], dtype=np.float32)
            tr = np.array(found_markers[1], dtype=np.float32)
            br = np.array(found_markers[2], dtype=np.float32)
            bl = np.array(found_markers[3], dtype=np.float32)
            poly_ordered = np.array([tl, tr, br, bl])

            area = cv2.contourArea(poly_ordered)
            is_convex = cv2.isContourConvex(poly_ordered.astype(np.int32))
            if not is_convex or area < 0.10 * w * h:
                return False, None, f"Marqueurs ArUco non convexes ou aire trop faible ({area:.0f} px)"

            # Vérification de parallélogramme/rectangle
            d_top = np.linalg.norm(tr - tl)
            d_bot = np.linalg.norm(br - bl)
            d_left = np.linalg.norm(bl - tl)
            d_right = np.linalg.norm(br - tr)
            if min(d_top, d_bot) <= 1e-4 or min(d_left, d_right) <= 1e-4:
                return False, None, "Marqueurs dégénérés"

            ratio_tb = max(d_top, d_bot) / min(d_top, d_bot)
            ratio_lr = max(d_left, d_right) / min(d_left, d_right)
            if ratio_tb > 1.45 or ratio_lr > 1.45:
                return False, None, f"Quadrilatère ArUco déformé (ratio {ratio_tb:.2f}, {ratio_lr:.2f})"

            raw_centers_norm = [[float(p[0] / w), float(p[1] / h)] for p in (tl, tr, br, bl)]
            self.last_aruco_centers_norm = raw_centers_norm
            board_corners_norm = self._apply_inset_to_centers(raw_centers_norm)
            return True, board_corners_norm, f"4/4 marqueurs détectés (IDs: {detected_ids}) + Retrait {cfg['aruco_inset_pct']}%"

        if len(detected_ids) > 0:
            return False, None, f"{len(detected_ids)}/4 marqueur(s) détecté(s) (IDs: {detected_ids})"

        return False, None, "Aucun marqueur ArUco détecté"

    def refit_grid(self, raw_frame):
        """Recalcule (ou réinitialise) le quadrillage interne selon l'option avec/sans Hough."""
        if raw_frame is not None and self.detection_config.get("use_hough_refine", True):
            return self.fit_grid_lines(raw_frame)
        self.reset_grid()
        return False

    def detect_corners_hough(self, frame):
        """
        Détecte les 4 coins de la zone de jeu 8x8 par transformée de Hough, sans marqueur ArUco :
        toutes les lignes sont détectées, puis on cherche le quadrillage 8x8 dont les cases
        alternent noir/blanc (voir hough_board.py). L'orientation (a8 = case claire) est gardée
        la plus proche possible de la calibration précédente.
        """
        if frame is None:
            return False, None, "Aucune image reçue"
        h, w = frame.shape[:2]
        previous = None
        if self.is_calibrated and self.corners_norm:
            previous = [[p[0] * w, p[1] * h] for p in self.corners_norm]
        ok, corners_px, msg = hough_board.detect_board(frame, previous_corners_px=previous)
        if not ok:
            return False, None, msg
        corners_norm = [[float(x / w), float(y / h)] for x, y in corners_px]
        return True, corners_norm, msg

    def snap_corners_to_hough_intersections(self, frame, initial_corners_norm, max_dist_px=22):
        """
        Couplage ArUco + Hough : recale précisément chaque coin initial sur l'intersection
        réelle de lignes perpendiculaires Hough la plus proche.
        """
        if frame is None or not initial_corners_norm or len(initial_corners_norm) != 4:
            return initial_corners_norm, 0

        h, w = frame.shape[:2]
        pts_init = np.float32([[p[0] * w, p[1] * h] for p in initial_corners_norm])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=12)
        if lines is None:
            return initial_corners_norm, 0

        refined = np.copy(pts_init)
        snapped_count = 0

        for i in range(4):
            target = pts_init[i]
            h_segs, v_segs = [], []
            for l in lines:
                x1, y1, x2, y2 = l.flatten()
                deg = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
                d1 = np.hypot(x1 - target[0], y1 - target[1])
                d2 = np.hypot(x2 - target[0], y2 - target[1])
                if min(d1, d2) < 45:
                    if deg < 30:
                        h_segs.append((x1, y1, x2, y2))
                    elif deg > 60:
                        v_segs.append((x1, y1, x2, y2))

            inter_candidates = []
            for x1, y1, x2, y2 in h_segs:
                dx1, dy1 = x2 - x1, y2 - y1
                for x3, y3, x4, y4 in v_segs:
                    dx2, dy2 = x4 - x3, y4 - y3
                    det = -dy1 * dx2 + dx1 * dy2
                    if abs(det) < 1e-3: continue
                    c1 = x1 * dy1 - y1 * dx1
                    c2 = x3 * dy2 - y3 * dx2
                    xi = (c1 * dx2 - dx1 * c2) / (-det)
                    yi = (c1 * dy2 - dy1 * c2) / (-det)
                    d = np.hypot(xi - target[0], yi - target[1])
                    if d < max_dist_px:
                        inter_candidates.append(([xi, yi], d))

            if inter_candidates:
                inter_candidates.sort(key=lambda item: item[1])
                refined[i] = inter_candidates[0][0]
                snapped_count += 1

        refined_norm = [[float(p[0] / w), float(p[1] / h)] for p in refined]
        return refined_norm, snapped_count

    def auto_detect_board(self, raw_frame):
        """
        Détection automatique selon le mode choisi :
        - 'COUPLED' : ArUco pour le contour externe + Hough pour l'alignement précis du quadrillage interne (Recommandé)
        - 'ARUCO'   : ArUco seul (contour externe ArUco + grille uniforme 8x8 standard)
        - 'HOUGH'   : Hough seul (lignes du damier sans marqueurs)
        """
        if raw_frame is None:
            return False, "Aucune image reçue", None

        method = self.detection_config.get("detection_method", "COUPLED").upper()

        if method == "HOUGH":
            ok, corners, msg = self.detect_corners_hough(raw_frame)
            if ok:
                self.set_corners(corners)
                if self.refit_grid(raw_frame):
                    msg += f" + Quadrillage ({self.refined_grid_count}/14 lignes)"
                return True, msg, corners

        elif method == "ARUCO":
            # Mode ArUco pur : 4 coins déterminés par ArUco + grille uniforme 8x8
            has_aruco, aruco_corners, aruco_msg = self.detect_aruco_corners(raw_frame)
            if has_aruco:
                self.set_corners(aruco_corners)
                self.reset_grid()
                return True, aruco_msg + " (Grille uniforme 8x8)", aruco_corners

        else: # 'COUPLED' (ArUco pour le contour + Hough pour le quadrillage interne)
            has_aruco, aruco_corners, aruco_msg = self.detect_aruco_corners(raw_frame)
            if has_aruco:
                self.set_corners(aruco_corners)
                if self.fit_grid_lines(raw_frame):
                    aruco_msg += f" + Quadrillage Hough ({self.refined_grid_count}/14 lignes)"
                return True, aruco_msg, aruco_corners

            # Fallback Hough pure si ArUco non détecté
            ok_hough, hough_corners, hough_msg = self.detect_corners_hough(raw_frame)
            if ok_hough:
                self.set_corners(hough_corners)
                if self.fit_grid_lines(raw_frame):
                    hough_msg += f" + Quadrillage Hough ({self.refined_grid_count}/14 lignes)"
                return True, "ArUco non détecté -> " + hough_msg, hough_corners

        h, w = raw_frame.shape[:2]
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Stratégie 1: findChessboardCornersSB sur les coins internes 7x7
        found = False
        corners = None
        try:
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCornersSB(gray_eq, (7, 7), flags)
        except Exception:
            pass

        if not found:
            try:
                found, corners = cv2.findChessboardCorners(gray, (7, 7), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            except Exception:
                pass

        if found and corners is not None and len(corners) == 49:
            # Grille interne 7x7 correspond aux coordonnées 64 à 448 dans 512x512
            inner_dst = []
            for r in range(1, 8):
                for c in range(1, 8):
                    inner_dst.append([c * SQUARE_SIZE, r * SQUARE_SIZE])
            inner_dst = np.float32(inner_dst)
            inner_src = corners.reshape(-1, 2).astype(np.float32)

            H_est, _ = cv2.findHomography(inner_dst, inner_src)
            if H_est is not None:
                outer_dst = np.float32([[[0, 0]], [[WARP_SIZE, 0]], [[WARP_SIZE, WARP_SIZE]], [[0, WARP_SIZE]]])
                projected_outer = cv2.perspectiveTransform(outer_dst, H_est).reshape(4, 2)
                norm_corners = [[float(pt[0] / w), float(pt[1] / h)] for pt in projected_outer]
                self.set_corners(norm_corners)
                return True, "Détection automatique réussie (Grille 7x7)", norm_corners

        # Stratégie 2: Fallback contours quadrilatère dominant
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = w * h
        best_quad = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.12 * img_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx) and area > max_area:
                    max_area = area
                    best_quad = approx.reshape(4, 2)

        if best_quad is not None:
            # Ordonner sommets: Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche
            pts = best_quad.astype(np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]

            ordered = [tl, tr, br, bl]
            norm_corners = [[float(pt[0] / w), float(pt[1] / h)] for pt in ordered]
            self.set_corners(norm_corners)
            return True, "Détection automatique par contour de plateau", norm_corners

        return False, "Aucun échiquier détecté automatiquement. Ajustez les 4 coins manuellement.", None

    def process_frame(self, frame):
        """Hook d'acquisition appelé sur chaque frame brute de CameraManager."""
        h, w = frame.shape[:2]
        with self._lock:
            self._update_homography(w, h)

            # 1. Redressement de perspective avec marge PASTILLE_PAD (64px) pour capturer les pastilles qui débordent du quadrillage
            pad_size = WARP_SIZE + 2 * PASTILLE_PAD
            warped_padded = cv2.warpPerspective(frame, self.H_pad, (pad_size, pad_size))
            self._latest_warped_padded = warped_padded
            warped = warped_padded[PASTILLE_PAD:PASTILLE_PAD + WARP_SIZE, PASTILLE_PAD:PASTILLE_PAD + WARP_SIZE]
            self._latest_warped_frame = warped.copy()

            # 2. Détection de stabilité et mouvement
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)

            if self._prev_warped_gray is not None:
                diff = cv2.absdiff(gray_warped, self._prev_warped_gray)
                self.motion_ratio = float(np.mean(diff > 18))
                now = time.time()
                if self.motion_ratio > 0.008:
                    self.is_board_stable = False
                    self.last_motion_time = now
                elif (now - self.last_motion_time) >= self.stability_delay:
                    self.is_board_stable = True
            self._prev_warped_gray = gray_warped.copy()

            # 3. Préparer affichage miniature warped 512x512
            warped_display = warped.copy()
            if self.show_warped_grid:
                self._draw_warped_grid_overlay(warped_display)
            if self.show_warped_arrow and self.ar_arrow:
                self._draw_warped_arrow(warped_display, self.ar_arrow)

            ok, buf = cv2.imencode('.jpg', warped_display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self._latest_warped_jpeg = buf.tobytes()

            # 4. Rendu Réalité Augmentée sur la caméra
            display_frame = frame.copy()

            # Mode "Vision Détection Ordinateur" : projection de la carte de gradient / contours
            if self.show_ar_vision_debug and self.is_calibrated and self.H_inv is not None:
                try:
                    lg = np.log(cv2.GaussianBlur(gray_warped.astype(np.float32), (3, 3), 0) + 8.0)
                    gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3)
                    lgrad = np.sqrt(gx * gx + gy * gy)
                    grad_norm = np.clip(lgrad * 125, 0, 255).astype(np.uint8)
                    grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_MAGMA)
                    unwarped_grad = cv2.warpPerspective(grad_color, self.H_inv, (w, h))

                    corners_px = np.array([(int(p[0] * w), int(p[1] * h)) for p in self.corners_norm], dtype=np.int32)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, corners_px, 255)
                    idx = mask > 0
                    display_frame[idx] = cv2.addWeighted(display_frame[idx], 0.15, unwarped_grad[idx], 0.85, 0)
                except Exception:
                    pass

            self._ar_frame_count += 1
            if self.show_ar_pieces and (not self.last_detected_pastilles or self._ar_frame_count % 3 == 0):
                try:
                    p_center = self.get_perspective_center()
                    fresh_pastilles = detect_pastilles_by_square(warped_padded, center_xy=p_center, pad=PASTILLE_PAD)
                    if fresh_pastilles:
                        self.last_detected_pastilles = fresh_pastilles
                except Exception:
                    pass

            self._draw_ar_elements(display_frame, w, h)

            return display_frame

    def _draw_warped_grid_overlay(self, img):
        # Grille active : alignée (avec Hough) ou uniforme (sans Hough). Le contour (lignes 0 et 8) reste fixe.
        g = self._active_grid()
        refined = g is getattr(self, 'refined_grid', None)
        line_col = (0, 220, 255) if refined else (0, 255, 120)
        for k in range(9):
            v = np.array([self.grid_point(k, r, g) for r in range(9)], dtype=np.float64)
            hl = np.array([self.grid_point(c, k, g) for c in range(9)], dtype=np.float64)
            for arr in (v, hl):
                cv2.polylines(img, [np.round(arr * 16).astype(np.int32)], False, line_col, 1, cv2.LINE_AA, shift=4)

        # Étiquettes colonnes / rangées centrées dans les cases réelles
        for c in range(8):
            cx, _ = self.square_center_warped(c, 7)
            cv2.putText(img, chr(ord('a') + c), (int(cx) - 4, WARP_SIZE - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for r in range(8):
            _, cy = self.square_center_warped(0, r)
            cv2.putText(img, str(8 - r), (6, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Hough ON" if refined else "Hough OFF", (WARP_SIZE - 92, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_col, 1, cv2.LINE_AA)

    def _draw_ar_elements(self, display_frame, w, h):
        """Dessine les overlays AR projetés sur la vidéo réelle."""
        overlay = display_frame.copy()

        # A. Guide de placement initial des pièces (rangées 1-2 et 7-8)
        if self.setup_mode:
            # Rangées 7 et 8 (Noirs, row 0 et 1 dans 512x512)
            pts_black = np.array(self.warped_points_to_camera([(0, 0), (WARP_SIZE, 0), (WARP_SIZE, 2 * SQUARE_SIZE), (0, 2 * SQUARE_SIZE)]), dtype=np.int32)
            cv2.fillPoly(overlay, [pts_black], (220, 100, 50))  # Bleu-nuit / orange
            # Rangées 1 et 2 (Blancs, row 6 et 7 dans 512x512)
            pts_white = np.array(self.warped_points_to_camera([(0, 6 * SQUARE_SIZE), (WARP_SIZE, 6 * SQUARE_SIZE), (WARP_SIZE, WARP_SIZE), (0, WARP_SIZE)]), dtype=np.int32)
            cv2.fillPoly(overlay, [pts_white], (80, 220, 100))  # Vert
            cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
            overlay = display_frame.copy()

        # B. Surbrillance de cases spécifiques (dernier coup, échec, sélection)
        if self.highlight_squares:
            for sq, info in self.highlight_squares.items():
                col = sq % 8
                row = 7 - (sq // 8)
                poly = self.square_to_camera_polygon(col, row)
                color = info.get("color", (0, 255, 255))
                alpha = info.get("alpha", 0.4)
                cv2.fillPoly(overlay, [poly], color)
                cv2.addWeighted(overlay, alpha, display_frame, 1.0 - alpha, 0, display_frame)
                overlay = display_frame.copy()

        # C. Affichage visuel des marqueurs ArUco détectés
        if self.show_aruco_debug and self.last_detected_markers:
            for mid, m_pts in self.last_detected_markers.items():
                pts = m_pts.astype(int)
                # Contour vert vif épais
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                # Centre et label ID
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(display_frame, f"ArUco #{mid}", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # D. Grille AR et contour perspective
        if self.show_ar_grid:
            g = self._active_grid()
            refined = g is not None and g is getattr(self, 'refined_grid', None)
            line_col = (0, 220, 130) if not refined else (255, 220, 0)  # cyan = quadrillage aligné
            for k in range(1, 8):
                v_pts = self.warped_points_to_camera([self.grid_point(k, r, g) for r in range(9)])
                h_pts = self.warped_points_to_camera([self.grid_point(c, k, g) for c in range(9)])
                for pts in (v_pts, h_pts):
                    arr = np.round(np.array(pts, dtype=np.float64) * 16).astype(np.int32)
                    cv2.polylines(display_frame, [arr], False, line_col, 1, cv2.LINE_AA, shift=4)

            # Cadre extérieur avec couleur d'orientation (Vert = Blancs / Rangée 1)
            corners_px = [(int(p[0] * w), int(p[1] * h)) for p in self.corners_norm]
            # Rangée 8 (Noirs): entre coin 0 et 1
            cv2.line(display_frame, corners_px[0], corners_px[1], (255, 130, 70), 2, cv2.LINE_AA)
            # Colonne h: entre 1 et 2
            cv2.line(display_frame, corners_px[1], corners_px[2], (0, 200, 200), 2, cv2.LINE_AA)
            # Rangée 1 (Blancs): entre coin 3 et 2
            cv2.line(display_frame, corners_px[3], corners_px[2], (80, 255, 100), 3, cv2.LINE_AA)
            # Colonne a: entre 3 et 0
            cv2.line(display_frame, corners_px[3], corners_px[0], (0, 200, 200), 2, cv2.LINE_AA)

            if self.show_ar_labels:
                # 1. Nom de chaque case (a1..h8) au centre selon la vision IA
                for col in range(8):
                    for row in range(8):
                        sq_name = f"{chr(ord('a') + col)}{8 - row}"
                        poly = self.square_to_camera_polygon(col, row)
                        cx = int(np.mean(poly[:, 0]))
                        cy = int(np.mean(poly[:, 1]))
                        # Contour noir + texte jaune fluo pour lisibilité parfaite
                        cv2.putText(display_frame, sq_name, (cx - 10, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(display_frame, sq_name, (cx - 10, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

                # 2. Lettres des colonnes A..H en bordure basse (Blancs) et haute (Noirs)
                for col in range(8):
                    letter = chr(ord('a') + col).upper()
                    # Bord bas (Blancs, row 7)
                    poly_b = self.square_to_camera_polygon(col, 7)
                    bx = int(np.mean(poly_b[:, 0]))
                    by = int(np.max(poly_b[:, 1])) + 16
                    cv2.putText(display_frame, letter, (bx - 5, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, letter, (bx - 5, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 100), 1, cv2.LINE_AA)

                    # Bord haut (Noirs, row 0)
                    poly_t = self.square_to_camera_polygon(col, 0)
                    tx = int(np.mean(poly_t[:, 0]))
                    ty = int(np.min(poly_t[:, 1])) - 8
                    cv2.putText(display_frame, letter, (tx - 5, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, letter, (tx - 5, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 130, 70), 1, cv2.LINE_AA)

                # 3. Chiffres des rangées 1..8 en bordure gauche (col a) et droite (col h)
                for row in range(8):
                    digit = str(8 - row)
                    # Bord gauche (colonne a)
                    poly_l = self.square_to_camera_polygon(0, row)
                    lx = int(np.min(poly_l[:, 0])) - 15
                    ly = int(np.mean(poly_l[:, 1])) + 5
                    cv2.putText(display_frame, digit, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, digit, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

                    # Bord droit (colonne h)
                    poly_r = self.square_to_camera_polygon(7, row)
                    rx = int(np.max(poly_r[:, 0])) + 6
                    ry = int(np.mean(poly_r[:, 1])) + 5
                    cv2.putText(display_frame, digit, (rx, ry),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, digit, (rx, ry),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

                # 4. Labels d'orientation générale
                mid_white = ((corners_px[3][0] + corners_px[2][0]) // 2 - 60, (corners_px[3][1] + corners_px[2][1]) // 2 + 35)
                cv2.putText(display_frame, "BLANCS (1)", mid_white, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 100), 2, cv2.LINE_AA)
                mid_black = ((corners_px[0][0] + corners_px[1][0]) // 2 - 50, (corners_px[0][1] + corners_px[1][1]) // 2 - 25)
                cv2.putText(display_frame, "NOIRS (8)", mid_black, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 130, 70), 2, cv2.LINE_AA)

        # D. Flèche AR projetée en perspective (coup du bot ou coach)
        if self.show_ar_arrow and self.ar_arrow:
            self._draw_perspective_arrow(display_frame, overlay, self.ar_arrow)

        # E. Pastilles des pièces détectées en direct avec lissage temporel anti-scintillement
        if self.show_ar_pieces:
            try:
                current_detections = {}

                # 1. Pastilles physiques de couleur détectées (Violet, Bleu, Rouge, Vert)
                # Projection projective exacte via H_inv : ancrée directement sur le sommet de la pièce
                if self.last_detected_pastilles and self.H_inv is not None:
                    p_items = list(self.last_detected_pastilles.items())
                    if p_items:
                        warp_pts = np.array([[[p.cx, p.cy]] for sq, p in p_items], dtype=np.float32)
                        cam_pts = cv2.perspectiveTransform(warp_pts, self.H_inv)
                        for i, (sq, p) in enumerate(p_items):
                            pt_cam = cam_pts[i][0]
                            is_w = p.type in ('PAWN_W', 'PIECE_W')
                            current_detections[sq] = {
                                'cam_x': float(pt_cam[0]),
                                'cam_y': float(pt_cam[1]),
                                'bgr': p.color_bgr,
                                'type': p.type,
                                'piece_color': 'white' if is_w else 'black',
                                'is_pastille': True,
                            }

                # 2. Mode Hybride (Pastilles + Pièces validées par le jeu)
                piece_mode = self.detection_config.get("piece_mode", "PASTILLES_STRICT")
                board = getattr(self, 'current_game_board', None)
                raw_p = self.last_detected_pieces

                if piece_mode == "HYBRID" and board is not None:
                    for sq in range(64):
                        if sq in current_detections:
                            continue  # Déjà couvert par une pastille physique de couleur
                        piece = board.piece_at(sq)
                        if piece is None:
                            continue  # Jamais de pièce fantôme sur une case vide

                        # Si la vision affirme que la case est vide ou que la pièce attendue n'y est plus :
                        if raw_p is not None and len(raw_p) > sq:
                            p_empty, p_white, p_black = raw_p[sq][:3]
                            if p_empty >= 0.45:
                                continue
                            if piece.color == chess.WHITE and p_white < 0.35:
                                continue
                            if piece.color == chess.BLACK and p_black < 0.35:
                                continue

                        col = sq % 8
                        row = 7 - (sq // 8)
                        poly = self.square_to_camera_polygon(col, row)
                        cx = float(np.mean(poly[:, 0]))
                        cy = float(np.mean(poly[:, 1]))
                        is_w = (piece.color == chess.WHITE)
                        current_detections[sq] = {
                            'cam_x': cx,
                            'cam_y': cy,
                            'bgr': (240, 240, 240) if is_w else (35, 35, 35),
                            'type': 'PIECE_W' if is_w else 'PIECE_B',
                            'piece_color': 'white' if is_w else 'black',
                            'is_pastille': False,
                        }

                # 3. Suivi temporel persistant (Debounce + Hystérésis anti-scintillement)
                if not hasattr(self, '_ar_piece_tracker') or self._ar_piece_tracker is None:
                    self._ar_piece_tracker = {}

                TTL_INIT = 8  # Persistance de ~260ms à 30fps

                for sq, det in current_detections.items():
                    if sq in self._ar_piece_tracker:
                        track = self._ar_piece_tracker[sq]
                        track['ttl'] = TTL_INIT
                        track['seen_count'] += 1
                        track['cam_x'] = 0.80 * track['cam_x'] + 0.20 * det['cam_x']
                        track['cam_y'] = 0.80 * track['cam_y'] + 0.20 * det['cam_y']
                        track['bgr'] = det['bgr']
                        track['is_pastille'] = det['is_pastille']
                        track['piece_color'] = det['piece_color']
                        track['type'] = det['type']
                    else:
                        init_seen = 2 if (det['is_pastille'] or board is not None) else 1
                        self._ar_piece_tracker[sq] = {
                            'cam_x': det['cam_x'],
                            'cam_y': det['cam_y'],
                            'bgr': det['bgr'],
                            'type': det['type'],
                            'piece_color': det['piece_color'],
                            'is_pastille': det['is_pastille'],
                            'ttl': TTL_INIT,
                            'seen_count': init_seen,
                        }

                dead_sqs = []
                for sq, track in self._ar_piece_tracker.items():
                    if sq not in current_detections:
                        track['ttl'] -= 1
                        if track['ttl'] <= 0:
                            dead_sqs.append(sq)
                for sq in dead_sqs:
                    del self._ar_piece_tracker[sq]

                # 4. Rendu visuel propre et stable (Apple Design)
                for sq, track in self._ar_piece_tracker.items():
                    if track['seen_count'] < 2 and not track['is_pastille']:
                        continue

                    cx = int(round(track['cam_x']))
                    cy = int(round(track['cam_y']))
                    bgr = track['bgr']
                    is_pastille = track['is_pastille']
                    p_color = track['piece_color']

                    if is_pastille:
                        # Anneau externe noir de contraste
                        cv2.circle(display_frame, (cx, cy), 13, (15, 15, 15), -1, cv2.LINE_AA)
                        # Pastille de couleur (Violet, Bleu, Rouge, Vert)
                        cv2.circle(display_frame, (cx, cy), 11, bgr, -1, cv2.LINE_AA)
                        # Filet lumineux blanc
                        cv2.circle(display_frame, (cx, cy), 12, (255, 255, 255), 1, cv2.LINE_AA)
                        # Point contrasté central
                        dot_col = (255, 255, 255) if p_color == 'black' else (20, 20, 20)
                        cv2.circle(display_frame, (cx, cy), 3, dot_col, -1, cv2.LINE_AA)
                    else:
                        if p_color == 'white':
                            cv2.circle(display_frame, (cx, cy), 12, (245, 245, 245), -1, cv2.LINE_AA)
                            cv2.circle(display_frame, (cx, cy), 12, (30, 30, 30), 2, cv2.LINE_AA)
                            cv2.circle(display_frame, (cx, cy), 4, (30, 30, 30), -1, cv2.LINE_AA)
                        else:
                            cv2.circle(display_frame, (cx, cy), 12, (35, 35, 35), -1, cv2.LINE_AA)
                            cv2.circle(display_frame, (cx, cy), 12, (240, 240, 240), 2, cv2.LINE_AA)
                            cv2.circle(display_frame, (cx, cy), 4, (240, 240, 240), -1, cv2.LINE_AA)
            except Exception:
                pass

        # F. Indicateur de stabilité en bandeau discret en haut
        status_color = (60, 210, 80) if self.is_board_stable else (0, 160, 255)
        status_text = "PLATEAU STABLE" if self.is_board_stable else "MOUVEMENT DETECTE..."
        cv2.circle(display_frame, (18, 22), 6, status_color, -1, cv2.LINE_AA)
        cv2.putText(display_frame, status_text, (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        if self.top_diff_info:
            cv2.putText(display_frame, self.top_diff_info, (200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_perspective_arrow(self, display_frame, overlay, arrow_data):
        from_sq = arrow_data.get("from_sq")
        to_sq = arrow_data.get("to_sq")
        if from_sq is None or to_sq is None:
            return

        is_capture = arrow_data.get("is_capture", False)
        color = arrow_data.get("color", (0, 220, 255))

        f_col = from_sq % 8
        f_row = 7 - (from_sq // 8)
        t_col = to_sq % 8
        t_row = 7 - (to_sq // 8)

        # Calcul dans le repère plat 512x512 (centres des cases réelles)
        fx, fy = self.square_center_warped(f_col, f_row)
        tx, ty = self.square_center_warped(t_col, t_row)

        # Surbrillance des 2 cases
        poly_from = self.square_to_camera_polygon(f_col, f_row)
        poly_to = self.square_to_camera_polygon(t_col, t_row)
        cv2.fillPoly(overlay, [poly_from], (0, 160, 255))
        cv2.fillPoly(overlay, [poly_to], (40, 220, 80) if not is_capture else (30, 40, 230))
        cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0, display_frame)

        # Flèche géométrique en 512x512
        dx = tx - fx
        dy = ty - fy
        length = np.hypot(dx, dy)
        if length > 5:
            ux = dx / length
            uy = dy / length
            vx = -uy
            vy = ux

            shaft_w = 7.0
            head_len = min(26.0, length * 0.4)
            head_w = 16.0

            base_head_x = tx - ux * head_len
            base_head_y = ty - uy * head_len

            p1 = (fx + vx * shaft_w, fy + vy * shaft_w)
            p2 = (base_head_x + vx * shaft_w, base_head_y + vy * shaft_w)
            p3 = (base_head_x + vx * head_w, base_head_y + vy * head_w)
            p4 = (tx, ty)
            p5 = (base_head_x - vx * head_w, base_head_y - vy * head_w)
            p6 = (base_head_x - vx * shaft_w, base_head_y - vy * shaft_w)
            p7 = (fx - vx * shaft_w, fy - vy * shaft_w)

            pts_warped = [p1, p2, p3, p4, p5, p6, p7]
            pts_cam = np.array(self.warped_points_to_camera(pts_warped), dtype=np.int32)
            cv2.fillPoly(display_frame, [pts_cam], color)
            cv2.polylines(display_frame, [pts_cam], True, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_warped_arrow(self, warped_frame, arrow_data):
        """Dessine directement la flèche de coup dans le repère redressé 512x512."""
        from_sq = arrow_data.get("from_sq")
        to_sq = arrow_data.get("to_sq")
        if from_sq is None or to_sq is None:
            return

        is_capture = arrow_data.get("is_capture", False)
        color = arrow_data.get("color", (0, 220, 255))

        f_col = from_sq % 8
        f_row = 7 - (from_sq // 8)
        t_col = to_sq % 8
        t_row = 7 - (to_sq // 8)

        # Surbrillance des 2 cases sur la miniature
        overlay = warped_frame.copy()
        fx1, fy1, fx2, fy2 = self.square_to_warped_rect(f_col, f_row)
        tx1, ty1, tx2, ty2 = self.square_to_warped_rect(t_col, t_row)
        cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 160, 255), -1)
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (40, 220, 80) if not is_capture else (30, 40, 230), -1)
        cv2.addWeighted(overlay, 0.35, warped_frame, 0.65, 0, warped_frame)

        # Coordonnées des centres
        fx = (fx1 + fx2) / 2.0
        fy = (fy1 + fy2) / 2.0
        tx = (tx1 + tx2) / 2.0
        ty = (ty1 + ty2) / 2.0

        dx = tx - fx
        dy = ty - fy
        length = np.hypot(dx, dy)
        if length > 5:
            ux = dx / length
            uy = dy / length
            vx = -uy
            vy = ux

            shaft_w = 7.0
            head_len = min(26.0, length * 0.4)
            head_w = 16.0

            base_head_x = tx - ux * head_len
            base_head_y = ty - uy * head_len

            p1 = (fx + vx * shaft_w, fy + vy * shaft_w)
            p2 = (base_head_x + vx * shaft_w, base_head_y + vy * shaft_w)
            p3 = (base_head_x + vx * head_w, base_head_y + vy * head_w)
            p4 = (tx, ty)
            p5 = (base_head_x - vx * head_w, base_head_y - vy * head_w)
            p6 = (base_head_x - vx * shaft_w, base_head_y - vy * shaft_w)
            p7 = (fx - vx * shaft_w, fy - vy * shaft_w)

            pts = np.array([p1, p2, p3, p4, p5, p6, p7], dtype=np.int32)
            cv2.fillPoly(warped_frame, [pts], color)
            cv2.polylines(warped_frame, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Fonctions Slice 2 : Extraction et analyse différentielle des 64 cases ---

    def get_square_roi(self, warped_512, sq, margin=12):
        """Extrait la zone centrale (40x40 par défaut) d'une case pour isoler la base de la pièce."""
        col = sq % 8
        row = 7 - (sq // 8)
        x1, y1, x2, y2 = self.square_to_warped_rect(col, row)
        return warped_512[y1 + margin : y2 - margin, x1 + margin : x2 - margin]

    def compute_all_square_changes(self, ref_warped, cur_warped):
        """
        Compare les 64 cases entre l'image de référence et l'image actuelle.
        Combine luminance LAB, chroma LAB et variance laplacienne (texture/relief).
        """
        if ref_warped is None or cur_warped is None:
            return {sq: 0.0 for sq in range(64)}

        # Conversions globales 1 seule fois pour toute l'image (gain x5 à x10 CPU)
        ref_lab = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_lab = cv2.cvtColor(cur_warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        diff_l = np.abs(cur_lab[:, :, 0] - ref_lab[:, :, 0])
        diff_ab = np.mean(np.abs(cur_lab[:, :, 1:] - ref_lab[:, :, 1:]), axis=2)

        ref_gray = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_warped, cv2.COLOR_BGR2GRAY)
        lap_ref = cv2.Laplacian(ref_gray, cv2.CV_32F)
        lap_cur = cv2.Laplacian(cur_gray, cv2.CV_32F)

        margin = 12
        scores = {}
        for sq in range(64):
            col = sq % 8
            row = 7 - (sq // 8)
            y1, y2 = row * SQUARE_SIZE + margin, (row + 1) * SQUARE_SIZE - margin
            x1, x2 = col * SQUARE_SIZE + margin, (col + 1) * SQUARE_SIZE - margin

            dl = float(np.mean(diff_l[y1:y2, x1:x2]))
            dab = float(np.mean(diff_ab[y1:y2, x1:x2]))
            edge_ref = float(np.var(lap_ref[y1:y2, x1:x2]))
            edge_cur = float(np.var(lap_cur[y1:y2, x1:x2]))
            edge_diff = min(abs(edge_cur - edge_ref), 100.0)

            score = 0.6 * dl + 0.8 * dab + 0.15 * edge_diff
            scores[sq] = round(score, 2)

        return scores

    def compute_square_structure(self, ref_warped, cur_warped, squares):
        """
        Pour chaque case demandée : (ncc, std_ref, std_cur).
        ncc = corrélation normalisée (centrée) entre la case de référence et la case actuelle.
        Proche de 1 : même contenu, éventuellement assombri (ombre, lumière) ; une case
        réellement vidée donne une valeur basse, même si une pièce voisine déborde dessus.
        """
        out = {}
        if ref_warped is None or cur_warped is None:
            return out
        for sq in squares:
            a = self.get_square_roi(ref_warped, sq)
            b = self.get_square_roi(cur_warped, sq)
            if a.size == 0 or b.size == 0 or a.shape != b.shape:
                continue
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            a -= a.mean()
            b -= b.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            ncc = float((a * b).sum() / den) if den > 1e-6 else 0.0
            out[sq] = (ncc, float(a.std()), float(b.std()))
        return out

    def get_perspective_center(self):
        """Calcule le centre optique effectif (Cx, Cy) en fonction de l'asymétrie perspective des 4 coins."""
        if not self.is_calibrated or not self.corners_norm or len(self.corners_norm) != 4:
            return (WARP_SIZE / 2.0, WARP_SIZE / 2.0)
        pts = np.array(self.corners_norm, dtype=np.float32)
        tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        tilt_y = (w_bot - w_top) / max(w_bot, w_top, 1e-4)
        tilt_x = (h_right - h_left) / max(h_right, h_left, 1e-4)
        cx = (WARP_SIZE / 2.0) + tilt_x * (WARP_SIZE / 4.0)
        cy = (WARP_SIZE / 2.0) + tilt_y * (WARP_SIZE / 4.0)
        return (cx, cy)

    def get_latest_warped_frame(self):
        with self._lock:
            return self._latest_warped_frame.copy() if self._latest_warped_frame is not None else None

    def get_latest_warped_padded(self):
        with self._lock:
            p = getattr(self, '_latest_warped_padded', None)
            return p.copy() if p is not None else None

    def get_warped_jpeg(self):
        with self._lock:
            return self._latest_warped_jpeg

    def get_status(self):
        with self._lock:
            return {
                'is_calibrated': self.is_calibrated,
                'corners_norm': self.corners_norm,
                'settings': {
                    'show_ar_grid': self.show_ar_grid,
                    'show_ar_labels': self.show_ar_labels,
                    'show_ar_arrow': self.show_ar_arrow,
                    'show_ar_vision_debug': self.show_ar_vision_debug,
                    'show_ar_pieces': self.show_ar_pieces,
                    'show_warped_grid': self.show_warped_grid,
                    'show_warped_arrow': self.show_warped_arrow,
                    'show_aruco_debug': self.show_aruco_debug
                },
                'detection_config': self.detection_config,
                'detected_marker_ids': sorted(list(self.last_detected_markers.keys())),
                'detected_aruco_ids': sorted(list(self.last_detected_markers.keys())),
                'grid_refined_lines': int(self.refined_grid_count) if self.refined_grid else 0,
                'is_board_stable': self.is_board_stable,
                'continuous_detection': self.continuous_detection,
                'motion_ratio': round(self.motion_ratio, 4),
                'pastilles_detected_count': len(self.last_detected_pastilles)
            }
