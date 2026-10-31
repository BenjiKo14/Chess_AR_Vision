"""
Détection robuste des pastilles de couleur posées sur les pièces d'échecs :
- Pions noirs : pastilles VIOLETTES
- Pièces rang 1 noir (arrière-rangée noire : tours, cavaliers, fous, reine, roi) : pastilles BLEUES
- Pions blancs : pastilles ROUGES
- Pièces rang 8 blanc (arrière-rangée blanche : tours, cavaliers, fous, reine, roi) : pastilles VERTES

Avec compensation de parallaxe 3D :
Quand la caméra est inclinée, le sommet d'une pièce haute (7-9 cm) est projeté
plus loin du centre optique que sa base posée au sol. Ce module compense
ce décalage projectif pour associer avec certitude chaque pastille à sa case réelle.
"""

import cv2
import numpy as np
import chess

SQ = 64
WARP_SIZE = 512
PASTILLE_PAD = 64  # Marge d'une case tout autour du quadrillage 512x512 pour capter les pièces hautes débordant par l'angle caméra

# Identifiants de pastille
PASTILLE_NONE = None
PASTILLE_PAWN_B = "PAWN_B"    # Violet (Pion Noir)
PASTILLE_PIECE_B = "PIECE_B"  # Bleu (Pièce Noire)
PASTILLE_PAWN_W = "PAWN_W"    # Rouge (Pion Blanc)
PASTILLE_PIECE_W = "PIECE_W"  # Vert (Pièce Blanche)


class PastilleDetection:
    """Représente une pastille détectée sur le plateau."""
    def __init__(self, ptype, cx, cy, area, circularity, color_bgr, base_x=None, base_y=None):
        self.type = ptype
        self.cx = cx
        self.cy = cy
        self.area = area
        self.circularity = circularity
        self.color_bgr = color_bgr
        # Coordonnées estimées de la base de la pièce au sol (après compensation de parallaxe)
        self.base_x = base_x if base_x is not None else cx
        self.base_y = base_y if base_y is not None else cy

    def to_dict(self):
        return {
            'type': self.type,
            'cx': float(self.cx),
            'cy': float(self.cy),
            'base_x': float(self.base_x),
            'base_y': float(self.base_y),
            'area': float(self.area),
            'circularity': float(self.circularity),
            'color_bgr': self.color_bgr
        }


def detect_pastille_blobs(warped_img, center_xy=None, pad=None, config=None):
    """
    Extrait tous les blobs de pastilles de couleur sur une image redressée (512x512 ou 640x640 avec pad=64).
    Retourne une liste de PastilleDetection dont les coordonnées (cx, cy) et (base_x, base_y)
    sont exprimées dans le repère standard [0..512] du plateau.
    """
    if warped_img is None:
        return []

    cfg = config or {}
    min_area = float(cfg.get("pastille_min_area", 45))
    max_area = float(cfg.get("pastille_max_area", 320))
    min_circ = float(cfg.get("pastille_min_circ", 0.52))
    min_sat = float(cfg.get("pastille_min_sat", 40))
    parallax_scale = float(cfg.get("parallax_scale", 1.0))

    h, w = warped_img.shape[:2]
    if pad is None:
        pad = PASTILLE_PAD if (h == WARP_SIZE + 2 * PASTILLE_PAD and w == WARP_SIZE + 2 * PASTILLE_PAD) else 0
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    B = warped_img[:, :, 0].astype(np.float32)
    G = warped_img[:, :, 1].astype(np.float32)
    R = warped_img[:, :, 2].astype(np.float32)

    # Vérifier si des profils calibrés sur l'échiquier réel sont disponibles
    prof = {}
    try:
        from pathlib import Path
        import json
        p_file = Path("calibrated_pastille_profiles.json")
        if p_file.exists():
            with open(p_file, "r", encoding="utf-8") as f_prof:
                prof_data = json.load(f_prof)
                prof = prof_data.get("colors", {})
    except Exception:
        prof = {}

    # Masques couleurs calibrés dynamiquement (ou par défaut robuste)
    p_green = prof.get(PASTILLE_PIECE_W, {}) if prof.get(PASTILLE_PIECE_W, {}).get("count_found", 0) > 0 else {}
    g_h_min = min(p_green.get("h_min", 35), 35)
    g_h_max = max(p_green.get("h_max", 85), 85)
    g_s_min = max(float(p_green.get("s_min", min_sat)), min_sat)
    g_v_min = p_green.get("v_min", 50)
    m_green = (
        ((H >= g_h_min) & (H <= g_h_max) & (S >= g_s_min) & (V >= g_v_min)) &
        (G > R + 10) & (G > B + 8)
    )

    # 2. Rouge (Pion Blanc)
    p_red = prof.get(PASTILLE_PAWN_W, {}) if prof.get(PASTILLE_PAWN_W, {}).get("count_found", 0) > 0 else {}
    r_s_min = max(float(p_red.get("s_min", min_sat * 1.3)), min_sat)
    r_v_min = p_red.get("v_min", 65)
    m_red = (
        (((H <= 14) | (H >= 166)) & (S >= r_s_min) & (V >= r_v_min)) |
        ((R > 130) & (R > G + 35) & (R > B + 35) & (S >= min_sat))
    )

    # 3. Bleu (Pièce Majeure Noire)
    p_blue = prof.get(PASTILLE_PIECE_B, {}) if prof.get(PASTILLE_PIECE_B, {}).get("count_found", 0) > 0 else {}
    b_h_min = min(p_blue.get("h_min", 95), 95)
    b_h_max = max(p_blue.get("h_max", 130), 132)
    b_s_min = max(float(p_blue.get("s_min", 45)), min_sat)
    b_v_min = p_blue.get("v_min", 55)
    m_blue = (
        ((H >= b_h_min) & (H <= b_h_max) & (S >= b_s_min) & (V >= b_v_min)) &
        (B > R * 1.35) & (B > G + 10)
    )

    # 4. Violet (Pion Noir)
    p_viol = prof.get(PASTILLE_PAWN_B, {}) if prof.get(PASTILLE_PAWN_B, {}).get("count_found", 0) > 0 else {}
    v_h_min = min(p_viol.get("h_min", 132), 132)
    v_h_max = max(p_viol.get("h_max", 170), 170)
    v_s_min = max(float(p_viol.get("s_min", min_sat)), min_sat)
    v_v_min = p_viol.get("v_min", 55)
    m_violet = (
        ((H >= v_h_min) & (H <= v_h_max) & (S >= v_s_min) & (V >= v_v_min)) &
        (B > G * 1.20) & (R > G * 1.10)
    )

    categories = [
        (PASTILLE_PAWN_W, m_red, (0, 0, 255), 0.50),   # Pions blancs (hauteur modérée)
        (PASTILLE_PIECE_W, m_green, (0, 255, 0), 0.65), # Pièces blanches majeures (plus hautes)
        (PASTILLE_PIECE_B, m_blue, (255, 160, 0), 0.45), # Pièces noires majeures
        (PASTILLE_PAWN_B, m_violet, (255, 0, 255), 0.35), # Pions noirs
    ]

    blobs = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for ptype, raw_mask, disp_bgr, height_ratio in categories:
        mask_u8 = (raw_mask.astype(np.uint8) * 255)
        cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            # Filtre de taille réglable
            if min_area <= area <= max_area:
                peri = cv2.arcLength(c, True)
                circ = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
                # Filtre de circularité réglable (rejette reflets étirés et ombres)
                if circ >= min_circ:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = (M['m10'] / M['m00']) - pad
                        cy = (M['m01'] / M['m00']) - pad

                        # Modèle radial centré sur l'échiquier (perspective conique 3D) :
                        is_major = 'PIECE' in ptype
                        h_pix = 34.0 if is_major else 24.0

                        CX = center_xy[0] if center_xy else (WARP_SIZE / 2.0)
                        CY = center_xy[1] if center_xy else (WARP_SIZE / 2.0)
                        rx = (cx - CX) / (WARP_SIZE / 2.0)
                        ry = (cy - CY) / (WARP_SIZE / 2.0)

                        # Correction 2D bidirectionnelle modulée par parallax_scale
                        base_x = cx - rx * h_pix * 0.45 * parallax_scale
                        base_y = cy - ry * h_pix * 0.75 * parallax_scale

                        # Rejet des objets très éloignés hors plateau
                        if base_x < -25 or base_x > WARP_SIZE + 25 or base_y < -25 or base_y > WARP_SIZE + 25:
                            continue

                        blobs.append(PastilleDetection(
                            ptype, cx, cy, area, circ, disp_bgr,
                            base_x=base_x, base_y=base_y
                        ))

    return blobs


def detect_pastilles_by_square(warped_img, center_xy=None, pad=None, config=None):
    """
    Associe chaque case de l'échiquier (indexée par chess.Square 0..63) à sa pastille détectée,
    en utilisant les coordonnées au sol corrigées de la parallaxe.
    Retourne un dict: { sq: PastilleDetection }
    """
    blobs = detect_pastille_blobs(warped_img, center_xy=center_xy, pad=pad, config=config)
    square_pastilles = {}
    square_scores = {}

    for blob in blobs:
        # Col 0..7 correspond à file a..h
        col = int(np.clip(round(blob.base_x / SQ - 0.5), 0, 7))
        # Row 0..7 : row 0 est le haut de l'image (rank 8), row 7 est le bas (rank 1)
        row = int(np.clip(round(blob.base_y / SQ - 0.5), 0, 7))

        file = col
        rank = 7 - row
        sq = chess.square(file, rank)

        score = blob.area * (blob.circularity ** 2)
        if sq not in square_pastilles or score > square_scores[sq]:
            square_pastilles[sq] = blob
            square_scores[sq] = score

    return square_pastilles
