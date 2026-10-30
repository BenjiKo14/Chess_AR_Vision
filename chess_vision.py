import os
import sys
import json
import time
import threading
from pathlib import Path
import cv2
import numpy as np

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
        self.show_warped_grid = True

        # Stabilité et mouvement (MotionGate)
        self.is_board_stable = True
        self.motion_ratio = 0.0
        self.last_motion_time = time.time()
        self.stability_delay = 0.75

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
                self.save_calibration()
                return True
        return False

    def rotate_board_90(self):
        """Pivote circulairement les 4 coins pour réaligner les Blancs."""
        with self._lock:
            # Rotation horaire: A8 devient H8 etc.
            self.corners_norm = [self.corners_norm[1], self.corners_norm[2], self.corners_norm[3], self.corners_norm[0]]
            self.last_frame_shape = None
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

    def square_to_warped_rect(self, col, row):
        """Retourne (x1, y1, x2, y2) dans l'image 512x512 pour col in 0..7 (a..h) et row in 0..7 (8..1)."""
        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE
        return x1, y1, x2, y2

    def square_to_camera_polygon(self, col, row):
        """Retourne les 4 sommets projetés d'une case sur l'image caméra."""
        x1, y1, x2, y2 = self.square_to_warped_rect(col, row)
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return np.array(self.warped_points_to_camera(pts), dtype=np.int32)

    def auto_detect_board(self, raw_frame):
        """Détection automatique du plateau d'échecs via OpenCV."""
        if raw_frame is None:
            return False, "Aucune image reçue", None

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

            ok, buf = cv2.imencode('.jpg', warped_display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self._latest_warped_jpeg = buf.tobytes()

            # 4. Rendu Réalité Augmentée sur la caméra
            display_frame = frame.copy()
            self._draw_ar_elements(display_frame, w, h)

            return display_frame

    def _draw_warped_grid_overlay(self, img):
        for i in range(9):
            pos = i * SQUARE_SIZE
            cv2.line(img, (pos, 0), (pos, WARP_SIZE), (0, 255, 120), 1, cv2.LINE_AA)
            cv2.line(img, (0, pos), (WARP_SIZE, pos), (0, 255, 120), 1, cv2.LINE_AA)

        # Afficher étiquettes colonnes / rangées sur les bords
        for c in range(8):
            label = chr(ord('a') + c)
            cv2.putText(img, label, (c * SQUARE_SIZE + 24, WARP_SIZE - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        for r in range(8):
            label = str(8 - r)
            cv2.putText(img, label, (6, r * SQUARE_SIZE + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

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

        # C. Grille AR et contour perspective
        if self.show_ar_grid:
            # Lignes horizontales et verticales
            for i in range(1, 8):
                pos = i * SQUARE_SIZE
                p_h = self.warped_points_to_camera([(0, pos), (WARP_SIZE, pos)])
                cv2.line(display_frame, (int(p_h[0][0]), int(p_h[0][1])), (int(p_h[1][0]), int(p_h[1][1])), (0, 220, 130), 1, cv2.LINE_AA)
                p_v = self.warped_points_to_camera([(pos, 0), (pos, WARP_SIZE)])
                cv2.line(display_frame, (int(p_v[0][0]), int(p_v[0][1])), (int(p_v[1][0]), int(p_v[1][1])), (0, 220, 130), 1, cv2.LINE_AA)

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
        if self.ar_arrow:
            self._draw_perspective_arrow(display_frame, overlay, self.ar_arrow)

        # E. Indicateur de stabilité en bandeau discret en haut
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

        # Calcul dans le repère plat 512x512
        fx = (f_col + 0.5) * SQUARE_SIZE
        fy = (f_row + 0.5) * SQUARE_SIZE
        tx = (t_col + 0.5) * SQUARE_SIZE
        ty = (t_row + 0.5) * SQUARE_SIZE

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
                    'show_warped_grid': self.show_warped_grid
                },
                'is_board_stable': self.is_board_stable,
                'motion_ratio': round(self.motion_ratio, 4)
            }
