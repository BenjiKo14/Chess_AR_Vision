"""
Module d'auto-calibration des pièces et pastilles au départ d'une partie.

Au départ de la partie (position d'échecs initiale standard) :
- Rangée 1 (Blancs majeurs a1-h1) : pastilles VERTES (PIECE_W)
- Rangée 2 (Pions blancs a2-h2)   : pastilles ROUGES (PAWN_W)
- Rangées 3 à 6 (32 cases)        : CASES VIDES (16 blanches, 16 noires)
- Rangée 7 (Pions noirs a7-h7)    : pastilles VIOLETTES (PAWN_B)
- Rangée 8 (Noirs majeurs a8-h8)  : pastilles BLEUES (PIECE_B)

L'auto-calibration extrait les profils exacts de couleur (HSV et ratios BGR)
des pastilles réelles sous l'éclairage de la pièce, mesure la luminosité des cases
vides claires et sombres, et ajuste les seuils de détection sur-mesure.
"""

import cv2
import numpy as np
import chess
from pathlib import Path
import json
import time

from pastille_detector import (
    PASTILLE_PAD, WARP_SIZE, SQ,
    PASTILLE_PAWN_B, PASTILLE_PIECE_B,
    PASTILLE_PAWN_W, PASTILLE_PIECE_W
)

PROFILES_FILE = Path("calibrated_pastille_profiles.json")


class PieceAutoCalibrator:
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or print
        self.profiles = self.load_profiles()

    def log(self, msg):
        try:
            self.log_fn(msg)
        except Exception:
            pass

    def load_profiles(self):
        try:
            if PROFILES_FILE.exists():
                with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"[WARN] Erreur lecture profils pastilles: {e}")
        return {}

    def save_profiles(self, profiles):
        self.profiles = profiles
        try:
            with open(PROFILES_FILE, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2)
            self.log("💾 Profils de calibration des pastilles enregistrés dans calibrated_pastille_profiles.json")
        except Exception as e:
            self.log(f"[WARN] Erreur écriture profils: {e}")

    def calibrate_from_starting_board(self, warped_frame, warped_padded=None, center_xy=None):
        """
        Analyse l'image du plateau en position de départ standard pour apprendre les vraies
        couleurs des 4 types de pastilles et les caractéristiques des cases vides.
        Retourne (success, message, report_dict).
        """
        if warped_frame is None:
            return False, "Aucune image redressée disponible", {}

        h, w = warped_frame.shape[:2]
        pad = PASTILLE_PAD if (warped_padded is not None and warped_padded.shape[0] == WARP_SIZE + 2 * PASTILLE_PAD) else 0
        src_img = warped_padded if (pad > 0 and warped_padded is not None) else warped_frame

        hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        B = src_img[:, :, 0].astype(np.float32)
        G = src_img[:, :, 1].astype(np.float32)
        R = src_img[:, :, 2].astype(np.float32)

        CX = center_xy[0] if center_xy else (WARP_SIZE / 2.0)
        CY = center_xy[1] if center_xy else (WARP_SIZE / 2.0)

        # 1. Échantillonner les cases vides centrales (rangées 3 à 6, row in 2..5)
        empty_light_samples = []
        empty_dark_samples = []

        for row in range(2, 6):
            for col in range(8):
                is_light = (row + col) % 2 == 0
                x1 = pad + col * SQ + 14
                x2 = pad + (col + 1) * SQ - 14
                y1 = pad + row * SQ + 14
                y2 = pad + (row + 1) * SQ - 14
                patch_bgr = src_img[y1:y2, x1:x2]
                patch_hsv = hsv[y1:y2, x1:x2]
                if is_light:
                    empty_light_samples.append({
                        'bgr_mean': np.mean(patch_bgr, axis=(0, 1)).tolist(),
                        'sat_max': float(np.percentile(patch_hsv[:, :, 1], 95)),
                        'val_mean': float(np.mean(patch_hsv[:, :, 2]))
                    })
                else:
                    empty_dark_samples.append({
                        'bgr_mean': np.mean(patch_bgr, axis=(0, 1)).tolist(),
                        'sat_max': float(np.percentile(patch_hsv[:, :, 1], 95)),
                        'val_mean': float(np.mean(patch_hsv[:, :, 2]))
                    })

        # Saturation maximale naturelle du damier vide (pour ne pas confondre avec des pastilles)
        all_empty_sats = [s['sat_max'] for s in empty_light_samples + empty_dark_samples]
        noise_sat_threshold = float(np.percentile(all_empty_sats, 90)) if all_empty_sats else 25.0

        # 2. Échantillonner les 4 rangées de pastilles
        # Rangée 1 (row = 7) : VERT (Pièces majeures Blanches)
        # Rangée 2 (row = 6) : ROUGE (Pions Blancs)
        # Rangée 7 (row = 1) : VIOLET (Pions Noirs)
        # Rangée 8 (row = 0) : BLEU (Pièces majeures Noires)
        target_rows = [
            (PASTILLE_PIECE_W, 7, "VERT (Blancs Majeurs)", (35, 88), is_major := True),
            (PASTILLE_PAWN_W, 6, "ROUGE (Pions Blancs)", (0, 15), is_major := False),
            (PASTILLE_PAWN_B, 1, "VIOLET (Pions Noirs)", (130, 172), is_major := False),
            (PASTILLE_PIECE_B, 0, "BLEU (Noirs Majeurs)", (92, 132), is_major := True)
        ]

        calibrated_colors = {}
        total_found = 0

        for ptype, row, label, h_range, major_flag in target_rows:
            samples_h = []
            samples_s = []
            samples_v = []
            samples_area = []
            samples_circ = []

            h_pix = 34.0 if major_flag else 24.0

            for col in range(8):
                # Centre théorique de la base
                cx_base = col * SQ + 32
                cy_base = row * SQ + 32

                # Décalage parallaxe attendu vers le sommet
                rx = (cx_base - CX) / (WARP_SIZE / 2.0)
                ry = (cy_base - CY) / (WARP_SIZE / 2.0)
                cx_top = pad + cx_base + rx * h_pix * 0.45
                cy_top = pad + cy_base + ry * h_pix * 0.75

                # Fenêtre de recherche autour du sommet attendu (70x70 px)
                wx1 = max(0, int(cx_top - 35))
                wx2 = min(src_img.shape[1], int(cx_top + 35))
                wy1 = max(0, int(cy_top - 35))
                wy2 = min(src_img.shape[0], int(cy_top + 35))

                patch_hsv = hsv[wy1:wy2, wx1:wx2]
                patch_bgr = src_img[wy1:wy2, wx1:wx2]

                # Trouver le blob le plus saturé avec la teinte attendue
                h_ch = patch_hsv[:, :, 0]
                s_ch = patch_hsv[:, :, 1]
                v_ch = patch_hsv[:, :, 2]

                if ptype == PASTILLE_PAWN_W:
                    # Rouge : H proche de 0 ou proche de 180
                    mask_c = (((h_ch <= 16) | (h_ch >= 165)) & (s_ch >= 45) & (v_ch >= 60))
                else:
                    mask_c = ((h_ch >= h_range[0]) & (h_ch <= h_range[1]) & (s_ch >= 30) & (v_ch >= 45))

                cnts, _ = cv2.findContours((mask_c.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_cnt = None
                best_score = -1.0

                for c in cnts:
                    area = cv2.contourArea(c)
                    if 25 <= area <= 400:
                        peri = cv2.arcLength(c, True)
                        circ = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
                        if circ >= 0.40:
                            score = area * circ
                            if score > best_score:
                                best_score = score
                                best_cnt = c

                if best_cnt is not None:
                    area = cv2.contourArea(best_cnt)
                    peri = cv2.arcLength(best_cnt, True)
                    circ = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0

                    # Extraire les pixels intérieurs du contour
                    c_mask = np.zeros(patch_hsv.shape[:2], dtype=np.uint8)
                    cv2.drawContours(c_mask, [best_cnt], -1, 255, -1)
                    pix_h = h_ch[c_mask == 255]
                    pix_s = s_ch[c_mask == 255]
                    pix_v = v_ch[c_mask == 255]

                    if len(pix_h) >= 15:
                        samples_h.extend(pix_h.tolist())
                        samples_s.extend(pix_s.tolist())
                        samples_v.extend(pix_v.tolist())
                        samples_area.append(area)
                        samples_circ.append(circ)
                        total_found += 1

            if samples_h:
                med_h = float(np.median(samples_h))
                std_h = float(np.std(samples_h))
                med_s = float(np.median(samples_s))
                std_s = float(np.std(samples_s))
                med_v = float(np.median(samples_v))
                med_area = float(np.median(samples_area))
                med_circ = float(np.median(samples_circ))

                calibrated_colors[ptype] = {
                    'label': label,
                    'count_found': len(samples_area),
                    'h_med': round(med_h, 1),
                    'h_std': round(std_h, 1),
                    'h_min': max(0, int(med_h - 2.5 * max(std_h, 6.0))),
                    'h_max': min(180, int(med_h + 2.5 * max(std_h, 6.0))),
                    's_med': round(med_s, 1),
                    's_min': max(25, int(med_s - 2.0 * max(std_s, 10.0))),
                    'v_med': round(med_v, 1),
                    'v_min': max(35, int(med_v * 0.55)),
                    'area_med': round(med_area, 1),
                    'circ_med': round(med_circ, 2)
                }
            else:
                calibrated_colors[ptype] = {
                    'label': label,
                    'count_found': 0,
                    'error': 'Aucun blob identifié sur la rangée'
                }

        summary = {
            'calibrated_at': time.time(),
            'total_pastilles_found': total_found,
            'noise_sat_floor': round(noise_sat_threshold, 1),
            'colors': calibrated_colors
        }

        if total_found >= 8:
            self.save_profiles(summary)
        else:
            self.log(f"[INFO] Auto-calibration : seulement {total_found} pastilles trouvées (<8), profils existants conservés.")

        msg = f"🎯 Auto-calibration terminée : {total_found}/32 pastilles détectées et calibrées."
        self.log(msg)
        return True, msg, summary


auto_calibrator = PieceAutoCalibrator()
