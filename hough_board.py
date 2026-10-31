"""
Détection du damier 8x8 par transformée de Hough, sans marqueur.

Principe :
1. On détecte toutes les droites de l'image (Hough standard sur les contours).
2. On les sépare en deux familles de directions (les « verticales » et les « horizontales » du damier).
3. Dans chaque famille, on cherche 9 droites dont l'espacement suit une progression
   régulière en perspective (projective 1D). Les droites manquantes (masquées par une pièce,
   bord peu contrasté) sont extrapolées par le modèle.
4. Les intersections des droites retenues donnent une homographie damier -> image.
5. Chaque hypothèse est vérifiée par l'alternance noir/blanc des 64 cases : c'est ce qui
   élimine les lignes parasites (parquet, carton, bordure en bois) et les décalages d'une rangée.
"""
import warnings

import cv2
import numpy as np

MAX_LINES = 300          # droites Hough conservées (les plus votées)
FAMILY_SPREAD_DEG = 22   # écart angulaire max autour de la direction dominante (perspective)
MIN_CHECKER_SCORE = 0.35 # score d'alternance minimal pour accepter un damier


# --- 1. Détection des droites ---------------------------------------------------------

def _detect_lines(gray):
    h, w = gray.shape
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    thr = max(40, int(min(h, w) * 0.15))
    lines = cv2.HoughLines(edges, 1, np.pi / 360, thr)
    if lines is None:
        return np.zeros((0, 2), np.float64)
    return lines[:MAX_LINES, 0, :].astype(np.float64)  # (rho, theta), triées par votes


def _circ_dist_deg(a, b):
    """Distance entre deux directions de droites (modulo 180°)."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def _split_families(lines):
    """Sépare les droites en deux familles de directions quasi perpendiculaires."""
    deg = np.degrees(lines[:, 1])
    # Histogramme circulaire pondéré par le rang (les plus votées comptent plus)
    weights = 1.0 / (1.0 + np.arange(len(lines)) / 30.0)
    hist = np.zeros(180)
    for d, wgt in zip(deg, weights):
        hist[int(d) % 180] += wgt
    kernel = np.array([1, 2, 3, 2, 1], np.float64)
    smooth = np.array([sum(kernel[j] * hist[(i + j - 2) % 180] for j in range(5)) for i in range(180)])

    p1 = int(np.argmax(smooth))
    mask = _circ_dist_deg(np.arange(180), p1) > 45
    if not mask.any():
        return None
    p2 = int(np.argmax(np.where(mask, smooth, -1)))
    fam_a = lines[_circ_dist_deg(deg, p1) <= FAMILY_SPREAD_DEG]
    fam_b = lines[_circ_dist_deg(deg, p2) <= FAMILY_SPREAD_DEG]
    return fam_a, fam_b


# --- 2. Positions des droites le long d'une transversale -------------------------------

def _mean_normal(fam):
    # Moyenne des directions en angle doublé (theta et theta+pi décrivent la même droite)
    ang = 2 * fam[:, 1]
    m = 0.5 * np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return np.array([np.cos(m), np.sin(m)])


def _positions_on_transversal(fam, center, n):
    """Abscisse t de l'intersection de chaque droite avec la droite center + t*n."""
    nl = np.stack([np.cos(fam[:, 1]), np.sin(fam[:, 1])], axis=1)
    denom = nl @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (fam[:, 0] - nl @ center) / denom
    return t


def _merge_duplicates(fam, t, tol):
    """Fusionne les droites quasi identiques (garde la plus votée de chaque groupe)."""
    order = np.argsort(t)
    keep = []
    for idx in order:
        if not np.isfinite(t[idx]):
            continue
        dup = False
        for k in keep:
            if abs(t[k] - t[idx]) < tol and _circ_dist_deg(np.degrees(fam[k, 1]), np.degrees(fam[idx, 1])) < 4:
                dup = True
                if idx < k:  # rang plus faible = plus de votes
                    keep[keep.index(k)] = idx
                break
        if not dup:
            keep.append(idx)
    keep = sorted(keep, key=lambda i: t[i])
    return fam[keep], t[keep]


# --- 3. Recherche de 9 droites régulièrement espacées (projectif 1D) -------------------

def _fit_projective_1d(t, min_spacing):
    """
    Cherche les modèles t(u) = (a*u + b) / (c*u + 1) qui expliquent le plus de droites
    pour 9 indices consécutifs u0..u0+8. Retourne une liste de candidats
    [(score, {indice_0_8: index_droite})], du meilleur au moins bon.
    """
    n = len(t)
    us = np.arange(-9, 12, dtype=np.float64)
    cands = []
    seen = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 5)):
            d1 = t[j] - t[i]
            if d1 < min_spacing:
                continue
            for l in range(j + 1, min(n, j + 5)):
                d2 = t[l] - t[j]
                if d2 < min_spacing or not (0.7 < d2 / d1 < 1.43):
                    continue
                t0, t1, t2 = t[i], t[j], t[l]
                den = 2.0 * (t1 - t2)
                c = (t2 - 2 * t1 + t0) / den
                a = t1 - t0 + c * t1
                b = t0
                q = c * us + 1.0
                valid = q > 0.05
                pred = np.where(valid, (a * us + b) / np.where(valid, q, 1.0), np.nan)
                spacing = np.abs(np.gradient(pred))
                # Droite observée la plus proche de chaque position prédite
                dist = np.abs(pred[:, None] - t[None, :])
                nearest = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
                ndist = dist[np.arange(len(us)), nearest]
                matched = valid & (ndist < 0.18 * spacing)
                # Meilleure fenêtre de 9 indices consécutifs
                best = None
                for s in range(len(us) - 8):
                    win = slice(s, s + 9)
                    if not valid[win].all():
                        continue
                    cnt = int(matched[win].sum())
                    res = float(np.mean(ndist[win][matched[win]] / spacing[win][matched[win]])) if cnt else 1.0
                    sc = cnt - res
                    if best is None or sc > best[0]:
                        best = (sc, s)
                if best is None or best[0] < 5:
                    continue
                s = best[1]
                mapping = {k: int(nearest[s + k]) for k in range(9) if matched[s + k]}
                key = tuple(sorted(mapping.items()))
                if key in seen:
                    continue
                seen.append(key)
                cands.append((best[0], mapping))
    cands.sort(key=lambda c: c[0], reverse=True)
    return cands


# --- 4. Homographie et vérification de l'alternance noir/blanc -------------------------

def _intersect(l1, l2):
    r1, th1 = l1
    r2, th2 = l2
    A = np.array([[np.cos(th1), np.sin(th1)], [np.cos(th2), np.sin(th2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([r1, r2]))


def _lattice_homography(fam_a, map_a, fam_b, map_b):
    """Homographie (indice droite A, indice droite B) -> pixel, à partir des intersections."""
    src, dst = [], []
    for ka, ia in map_a.items():
        for kb, ib in map_b.items():
            p = _intersect(fam_a[ia], fam_b[ib])
            if p is not None:
                src.append([ka, kb])
                dst.append(p)
    if len(src) < 6:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 4.0)
    return H


# Points d'échantillonnage dans une case : près des coins, là où les pièces masquent rarement
_SAMPLE_OFFSETS = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.5, 0.5]])


def _cell_values(gray_blur, H, ka0, kb0, size=8):
    """Luminosité robuste (médiane de 5 points) de chaque case de la fenêtre [ka0, ka0+size)x[kb0, kb0+size)."""
    h, w = gray_blur.shape
    ka = np.arange(size)[:, None, None] + ka0
    kb = np.arange(size)[None, :, None] + kb0
    pa = ka + _SAMPLE_OFFSETS[None, None, :, 0]
    pb = kb + _SAMPLE_OFFSETS[None, None, :, 1]
    pts = np.stack([np.broadcast_to(pa, (size, size, 5)), np.broadcast_to(pb, (size, size, 5))], axis=-1)
    img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(size, size, 5, 2)
    x = np.round(img[..., 0]).astype(int)
    y = np.round(img[..., 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.mean() < 0.9:
        return None
    vals = gray_blur[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)].astype(np.float64)
    vals[~inside] = np.nan
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # case entièrement hors image
        return np.nanmedian(vals, axis=2)


def _checker_score(cells):
    """Alternance noir/blanc des 64 cases, entre -1 et 1 (signe : case (0,0) claire si > 0)."""
    if cells is None or np.isnan(cells).any():
        return 0.0
    parity = np.where((np.add.outer(np.arange(8), np.arange(8)) % 2) == 0, 1.0, -1.0)
    v = cells - cells.mean()
    norm = np.sqrt((v ** 2).sum()) * 8.0
    if norm < 1e-6:
        return 0.0
    return float((parity * v).sum() / norm)


# --- 5. Orientation des coins -----------------------------------------------------------

def _order_clockwise(pts):
    """Ordonne 4 points de l'image : haut-gauche, haut-droite, bas-droite, bas-gauche."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # sens horaire (axe y vers le bas)
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def _warped_parity(gray_blur, corners_px):
    """+1 si la case (0,0) du repère redressé (a8) est claire, -1 sinon."""
    dst = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    H = cv2.getPerspectiveTransform(dst, np.float32(corners_px))
    cells = _cell_values(gray_blur, H, 0, 0)
    return np.sign(_checker_score(cells))


def detect_board(frame, previous_corners_px=None):
    """
    Retourne (ok, corners_px, message). corners_px = 4 coins de la zone de jeu 8x8 dans l'ordre
    attendu par l'application (a8, h8, h1, a1), en pixels. La case a8 est toujours claire :
    il reste deux orientations possibles, on garde la plus proche de la calibration précédente.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lines = _detect_lines(gray)
    if len(lines) < 8:
        return False, None, f"Pas assez de lignes Hough ({len(lines)})"
    fams = _split_families(lines)
    if fams is None:
        return False, None, "Impossible de trouver deux directions de lignes"

    center = np.array([w / 2.0, h / 2.0])
    min_spacing = 0.025 * min(h, w)
    prepared = []
    for fam in fams:
        if len(fam) < 4:
            return False, None, f"Lignes insuffisantes dans une direction ({len(fam)})"
        n = _mean_normal(fam)
        t = _positions_on_transversal(fam, center, n)
        fam_m, t_m = _merge_duplicates(fam, t, tol=0.6 * min_spacing)
        cands = _fit_projective_1d(t_m, min_spacing)
        if not cands:
            return False, None, "Aucune suite de 9 lignes régulières trouvée"
        prepared.append((fam_m, cands[:6]))

    (fam_a, cands_a), (fam_b, cands_b) = prepared
    best = None
    for sa, map_a in cands_a:
        for sb, map_b in cands_b:
            H = _lattice_homography(fam_a, map_a, fam_b, map_b)
            if H is None:
                continue
            # Une droite de bord manquante peut décaler la fenêtre d'un cran : on teste ±1
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    cells = _cell_values(gray_blur, H, da, db)
                    score = abs(_checker_score(cells))
                    total = score + 0.01 * (sa + sb)
                    if best is None or total > best[0]:
                        best = (total, score, H, da, db, len(map_a) + len(map_b))

    if best is None or best[1] < MIN_CHECKER_SCORE:
        sc = 0.0 if best is None else best[1]
        return False, None, f"Damier 8x8 non reconnu (alternance noir/blanc {sc:.2f})"

    _, score, H, da, db, n_lines = best
    lattice = np.float32([[da, db], [da + 8, db], [da + 8, db + 8], [da, db + 8]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(lattice, H).reshape(4, 2)
    corners = _order_clockwise(corners)

    # Orientation : a8 (coin 0) doit être une case claire
    parity = _warped_parity(gray_blur, corners)
    options = [0, 2] if parity > 0 else [1, 3]
    rotations = [np.roll(corners, -r, axis=0) for r in options]
    if previous_corners_px is not None:
        prev = np.asarray(previous_corners_px, np.float64)
        corners = min(rotations, key=lambda c: np.linalg.norm(c - prev, axis=1).sum())
    else:
        corners = min(rotations, key=lambda c: c[0, 0] + c[0, 1])

    msg = f"Damier trouvé par Hough ({n_lines}/18 lignes vues, alternance noir/blanc {score:.2f})"
    return True, corners, msg
