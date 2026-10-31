import os
import sys
import json
import time
import threading
from pathlib import Path
import cv2
import numpy as np

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
        self._load_calibration()

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
        self.last_aruco_centers_norm = None  # Stocke les 4 centres bruts des marqueurs pour ajuster l'inset en direct
        self.last_detected_pieces = None    # Stocke les probabilités de présence par case [P_vide, P_blanc, P_noir]
        # Lignes directrices exactes détectées par Hough (9 lignes horizontales et 9 lignes verticales dans 512x512)
        # Quadrillage interne affiné (lignes internes calées sur les vraies séparations ; contour jamais modifié)
        self.refined_grid = None
        self.refined_grid_count = 0
        self.hough_grid_lines = {
            'rows': [float(i * SQUARE_SIZE) for i in range(9)],
            'cols': [float(j * SQUARE_SIZE) for j in range(9)]
        }

        # Console de configuration des paramètres de détection (réglables depuis l'UI Web)
        self.detection_config = {
            "min_side_px": 8,
            "max_side_px": 52,
            "clahe_clip": 3.5,
            "bin_thresh": 125,
            "border_err": 0.85,
            "err_corr": 0.85,
            "aruco_inset_pct": 2.4,  # Retrait automatique (%) des marqueurs vers les coins du damier 8x8
            "use_hough_refine": True, # Couplage avec la transformée de Hough pour calage subpixel sur les lignes de cases
            "detection_method": "COUPLED", # 'ARUCO', 'HOUGH', 'COUPLED'
        }

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
                        self.log("✅ Calibration plateau chargée depuis board_calibration.json")
        except Exception as e:
            self.log(f"⚠️ Erreur chargement calibration: {e}")

    def save_calibration(self):
        try:
            with open(self.calibration_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'corners': self.corners_norm,
                    'is_calibrated': self.is_calibrated,
                    'updated_at': time.time()
                }, f, indent=2)
            self.log("💾 Calibration sauvegardée.")
        except Exception as e:
            self.log(f"❌ Erreur sauvegarde calibration: {e}")

    def set_corners(self, corners):
        """Met à jour les 4 coins normalisés [[x0,y0], ... [x3,y3]]."""
        with self._lock:
            if len(corners) == 4:
                self.corners_norm = [[float(p[0]), float(p[1])] for p in corners]
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

    def update_detection_config(self, new_cfg, raw_frame=None):
        """Met à jour les paramètres de la console de détection et recalcule les coins si nécessaire."""
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
        Détecte les 4 marqueurs aux coins avec un pipeline 3 passes haute sensibilité :
        - Passe 1 : ArUco multi-échelle (x1.0 à x2.5) sur Gris & CLAHE (haute confiance)
        - Passe 2 : ArUco multi-échelle sur CLAHE + Binaire avec tolérance de bordure coupée
        - Passe 3 : Détecteur direct de petit carré noir à cœur blanc (CLAHE + Binaire) pour les marqueurs très petits (~10-15 px)
        """
        if frame is None:
            return False, None, "Pas d'image reçue"

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        cfg = self.detection_config
        min_side = float(cfg.get("min_side_px", 8))
        max_side = float(cfg.get("max_side_px", 52))
        clahe_clip = float(cfg.get("clahe_clip", 3.5))
        bin_thresh = int(cfg.get("bin_thresh", 125))
        border_err = float(cfg.get("border_err", 0.85))
        err_corr = float(cfg.get("err_corr", 0.85))

        found_markers = {}
        marker_polys = {}

        # Passe 1 : ArUco multi-échelle standard sur gray + clahe (0 faux positif)
        for scale in (1.0, 1.5, 2.0, 2.5):
            up = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(up)
            for src in (up, clahe):
                params = cv2.aruco.DetectorParameters()
                params.minMarkerPerimeterRate = 0.01
                params.maxMarkerPerimeterRate = 0.25
                params.polygonalApproxAccuracyRate = 0.06
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                det = cv2.aruco.ArucoDetector(d, params)
                corners, ids, _ = det.detectMarkers(src)
                if ids is not None:
                    for m, c in zip(ids.flatten(), corners):
                        mid = int(m)
                        if mid in (0, 1, 2, 3) and mid not in found_markers:
                            pts = c[0] / scale
                            side = float(np.linalg.norm(pts[0] - pts[1]))
                            if min_side <= side <= max_side:
                                center = pts.mean(axis=0)
                                if all(np.hypot(center[0] - v[0], center[1] - v[1]) > 50 for v in found_markers.values()):
                                    found_markers[mid] = (float(center[0]), float(center[1]))
                                    marker_polys[mid] = pts
            if len(found_markers) == 4:
                break

        # Passe 2 : Si < 4 marqueurs, passe CLAHE + Binaire + Tolérance de bordure (bords coupés aux ciseaux)
        if len(found_markers) < 4:
            for scale in (1.0, 1.5, 2.0):
                up = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(up)
                _, bin_clahe = cv2.threshold(clahe, bin_thresh, 255, cv2.THRESH_BINARY)
                for src in (clahe, bin_clahe):
                    params = cv2.aruco.DetectorParameters()
                    params.minMarkerPerimeterRate = 0.008
                    params.maxMarkerPerimeterRate = 0.25
                    params.adaptiveThreshWinSizeMin = 3
                    params.adaptiveThreshWinSizeMax = 45
                    params.adaptiveThreshWinSizeStep = 4
                    params.maxErroneousBitsInBorderRate = border_err
                    params.errorCorrectionRate = err_corr
                    params.polygonalApproxAccuracyRate = 0.07
                    det = cv2.aruco.ArucoDetector(d, params)
                    corners, ids, _ = det.detectMarkers(src)
                    if ids is not None:
                        for m, c in zip(ids.flatten(), corners):
                            mid = int(m)
                            if mid in (0, 1, 2, 3) and mid not in found_markers:
                                pts = c[0] / scale
                                side = float(np.linalg.norm(pts[0] - pts[1]))
                                if min_side <= side <= max_side:
                                    center = pts.mean(axis=0)
                                    if all(np.hypot(center[0] - v[0], center[1] - v[1]) > 60 for v in found_markers.values()):
                                        found_markers[mid] = (float(center[0]), float(center[1]))
                                        marker_polys[mid] = pts
                if len(found_markers) == 4:
                    break

        # Passe 3 : Détecteur direct de petit carré noir à cœur blanc (CLAHE + Binaire)
        if len(found_markers) < 4:
            cl_full = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(gray)
            for th_val in (bin_thresh, max(40, bin_thresh - 15), min(220, bin_thresh + 15), 100):
                _, binary_inv = cv2.threshold(cl_full, int(th_val), 255, cv2.THRESH_BINARY_INV)
                cnts, _ = cv2.findContours(binary_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x, y, bw, bh = cv2.boundingRect(c)
                    if min_side <= bw <= max_side and min_side <= bh <= max_side:
                        if 0.72 <= (bw / float(bh)) <= 1.38 and (cv2.contourArea(c) / float(bw * bh)) > 0.55:
                            patch = cl_full[y:y+bh, x:x+bw]
                            m = max(2, bw // 5)
                            inner = patch[m:-m, m:-m]
                            if inner.size > 0:
                                bmask = np.ones_like(patch, dtype=bool)
                                bmask[m:-m, m:-m] = False
                                if np.mean(patch[bmask]) < 125 and np.percentile(inner, 90) > 150 and np.std(inner) > 32:
                                    cx, cy = x + bw / 2.0, y + bh / 2.0
                                    if all(np.hypot(cx - v[0], cy - v[1]) > 60 for v in found_markers.values()):
                                        missing_ids = sorted(list(set([0, 1, 2, 3]) - set(found_markers.keys())))
                                        if missing_ids:
                                            mid = missing_ids[0]
                                            found_markers[mid] = (float(cx), float(cy))
                                            marker_polys[mid] = np.array([[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]], dtype=np.float32)
                                        if len(found_markers) == 4:
                                            break
                if len(found_markers) == 4:
                    break

        self.last_detected_markers = marker_polys
        detected_ids = sorted(list(found_markers.keys()))

        if len(found_markers) == 4:
            # Ordonner géométriquement les 4 marqueurs pour garantir Haut-Gauche, Haut-Droite, Bas-Droite, Bas-Gauche
            # ou utiliser l'ordre [0, 1, 2, 3] si tous sont cohérents
            pts_list = [found_markers[0], found_markers[1], found_markers[2], found_markers[3]] if all(k in found_markers for k in (0, 1, 2, 3)) else list(found_markers.values())
            pts_np = np.array(pts_list, dtype=np.float32)
            sums = pts_np.sum(axis=1)
            diffs = np.diff(pts_np, axis=1).flatten()

            tl = pts_np[np.argmin(sums)]
            br = pts_np[np.argmax(sums)]
            tr = pts_np[np.argmin(diffs)]  # x grand, y petit -> y - x minimal
            bl = pts_np[np.argmax(diffs)]  # x petit, y grand -> y - x maximal

            raw_centers_norm = [[float(p[0] / w), float(p[1] / h)] for p in (tl, tr, br, bl)]
            self.last_aruco_centers_norm = raw_centers_norm
            board_corners_norm = self._apply_inset_to_centers(raw_centers_norm)
            # Le contour reste celui des ArUco : l'alignement du quadrillage interne est fait ensuite
            # par fit_grid_lines() sans modifier ces 4 coins.
            return True, board_corners_norm, f"4/4 marqueurs detectes (IDs: {detected_ids}) + Retrait bordure {cfg['aruco_inset_pct']}%"

        if len(detected_ids) > 0:
            return False, None, f"{len(detected_ids)}/4 marqueur(s) detecte(s) (IDs: {detected_ids}) — ajustez les curseurs dans la Console Config"

        return False, None, "Aucun marqueur ArUco detecte — ajustez les curseurs CLAHE / Seuil Binaire"

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

            # 1. Redressement de perspective 512x512
            warped = cv2.warpPerspective(frame, self.H, (WARP_SIZE, WARP_SIZE))
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
                # Label côté Blancs
                mid_white = ((corners_px[3][0] + corners_px[2][0]) // 2 - 60, (corners_px[3][1] + corners_px[2][1]) // 2 + 20)
                cv2.putText(display_frame, "BLANCS (1)", mid_white, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 100), 2, cv2.LINE_AA)
                # Label côté Noirs
                mid_black = ((corners_px[0][0] + corners_px[1][0]) // 2 - 50, (corners_px[0][1] + corners_px[1][1]) // 2 - 10)
                cv2.putText(display_frame, "NOIRS (8)", mid_black, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 130, 70), 2, cv2.LINE_AA)

        # D. Flèche AR projetée en perspective (coup du bot ou coach)
        if self.show_ar_arrow and self.ar_arrow:
            self._draw_perspective_arrow(display_frame, overlay, self.ar_arrow)

        # E. Pastilles des pièces détectées en direct
        if self.show_ar_pieces and self.last_detected_pieces is not None:
            try:
                for sq in range(min(64, len(self.last_detected_pieces))):
                    p_empty, p_white, p_black = self.last_detected_pieces[sq]
                    col = sq % 8
                    row = 7 - (sq // 8)
                    poly = self.square_to_camera_polygon(col, row)
                    cx = int(np.mean(poly[:, 0]))
                    cy = int(np.mean(poly[:, 1]))

                    if p_white > 0.50:
                        cv2.circle(display_frame, (cx, cy), 12, (245, 245, 245), -1, cv2.LINE_AA)
                        cv2.circle(display_frame, (cx, cy), 12, (30, 30, 30), 2, cv2.LINE_AA)
                        cv2.circle(display_frame, (cx, cy), 4, (20, 20, 20), -1, cv2.LINE_AA)
                    elif p_black > 0.50:
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

        scores = {}
        for sq in range(64):
            roi_ref = self.get_square_roi(ref_warped, sq)
            roi_cur = self.get_square_roi(cur_warped, sq)

            if roi_ref.size == 0 or roi_cur.size == 0:
                scores[sq] = 0.0
                continue

            ref_lab = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2LAB)
            cur_lab = cv2.cvtColor(roi_cur, cv2.COLOR_BGR2LAB)

            diff_l = float(np.mean(np.abs(cur_lab[:, :, 0].astype(float) - ref_lab[:, :, 0].astype(float))))
            diff_ab = float(np.mean(np.abs(cur_lab[:, :, 1:].astype(float) - ref_lab[:, :, 1:].astype(float))))

            ref_gray = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(roi_cur, cv2.COLOR_BGR2GRAY)
            edge_ref = float(cv2.Laplacian(ref_gray, cv2.CV_64F).var())
            edge_cur = float(cv2.Laplacian(cur_gray, cv2.CV_64F).var())
            edge_diff = min(abs(edge_cur - edge_ref), 100.0)

            score = 0.6 * diff_l + 0.8 * diff_ab + 0.15 * edge_diff
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

    def get_latest_warped_frame(self):
        with self._lock:
            return self._latest_warped_frame.copy() if self._latest_warped_frame is not None else None

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
                'motion_ratio': round(self.motion_ratio, 4)
            }
