"""
Lecture de la position directement sur l'image redressée (512x512), sans image de référence.

Pour chaque case, on estime la probabilité qu'elle soit vide (E), occupée par une pièce
blanche (W) ou par une pièce noire (B), en deux étapes :

1. Occupée ou non : uniquement d'après les contours NETS, mesurés sur le logarithme de
   l'intensité. Une ombre assombrit une case (facteur multiplicatif, donc constant en
   logarithme) avec des bords flous : elle ne crée presque pas de contours nets,
   contrairement à une pièce. Une ombre ne peut donc pas « devenir » une pièce.
2. Si occupée, blanche ou noire : d'après la luminosité de la pièce RELATIVE à la case
   (une pièce noire est bien plus sombre que la case, une blanche plus claire ou
   modérément ombrée), ce qui reste valable sous le dégradé d'une lampe.

Le modèle est ré-appris sur chaque image à partir de la position connue de la partie :
il s'adapte tout seul à la lumière et à la caméra. Les quelques cases qui ont réellement
changé (un coup joué) sont mal étiquetées : apprentissage robuste (médianes) en deux
passes, la seconde sans les cases qui contredisent nettement leur étiquette.

INTÉGRATION PASTILLES DE COULEURS :
- Pions noirs : pastilles VIOLETTES
- Pièces rang 1 noir : pastilles BLEUES
- Pions blancs : pastilles ROUGES
- Pièces rang 8 blanc : pastilles VERTES
En présence d'une pastille clairement détectée sur une case, la probabilité de la
couleur correspondante est portée à >= 0.95.
"""
import cv2
import numpy as np
import chess
from pastille_detector import (
    detect_pastilles_by_square,
    PASTILLE_PAWN_B,
    PASTILLE_PIECE_B,
    PASTILLE_PAWN_W,
    PASTILLE_PIECE_W
)

SQ = 64          # taille d'une case dans l'image redressée
MARGIN = 10      # on ignore les bords de case (lignes du damier)
CLASSES = ("E", "W", "B")
P_MIN, P_MAX = 0.03, 0.97   # aucune case ne doit peser trop lourd à elle seule
# Planchers d'écart-type (évitent des modèles trop sûrs d'eux)
_OCC_FLOOR = np.array([0.05, 0.15, 0.15])  # part de contours nets, log du 90e centile, log de la moyenne
EDGE_LOG_GRAD = 0.25   # contour net en bonne lumière (gradient du log de l'intensité)
_COL_FLOOR = 10.0                     # luminosité relative de la pièce


def square_group(sq):
    """1 = case claire, 0 = case foncée (a1 est foncée)."""
    return (chess.square_rank(sq) + chess.square_file(sq)) % 2


def _square_slice(sq, margin=MARGIN):
    r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
    return slice(r * SQ + margin, r * SQ + SQ - margin), slice(c * SQ + margin, c * SQ + SQ - margin)


def _square_center(sq):
    return chess.square_file(sq) + 0.5, 7 - chess.square_rank(sq) + 0.5


def _label(board, sq):
    p = board.piece_at(sq)
    if p is None:
        return "E"
    return "W" if p.color == chess.WHITE else "B"


def _robust(rows, floor):
    rows = np.asarray(rows, np.float64)
    med = np.median(rows, axis=0)
    mad = 1.4826 * np.median(np.abs(rows - med), axis=0)
    return med, np.maximum(mad, floor)


def _student_ll(x, m, sd):
    """Log-vraisemblance à queues lourdes : une valeur aberrante ne décide pas seule."""
    z = (np.asarray(x) - m) / sd
    return float(np.sum(-np.log(sd) - 1.5 * np.log1p(z * z / 3.0)))


class BoardReader:
    def __init__(self):
        self.last_probs = None
        self.last_pastilles = {}

    # --- Caractéristiques ---------------------------------------------------------

    @staticmethod
    def _fit_empty_colour(lab_med, board, group, exclude=()):
        """Couleur d'une case vide du groupe selon sa position (dégradé de lampe)."""
        sqs = [s for s in chess.SQUARES if square_group(s) == group
               and board.piece_at(s) is None and s not in exclude]
        if not sqs:
            sqs = [s for s in chess.SQUARES if square_group(s) == group]
        X = np.array([[1.0, *_square_center(s)] for s in sqs])
        Y = np.array([lab_med[s] for s in sqs])
        if len(sqs) >= 6:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            res = np.linalg.norm(Y - X @ coef, axis=1)
            keep = res <= max(12.0, 2.5 * np.median(res))
            if keep.sum() >= 6:
                coef, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
            return lambda s: np.array([1.0, *_square_center(s)]) @ coef
        base = np.median(Y, axis=0)
        return lambda s: base

    def _features(self, warped, board, exclude=()):
        """occ[s] = (part de contours nets, log du 90e centile et log de la moyenne du gradient
        de log-intensité) : la première est très sûre en bonne lumière, les deux autres sans
        seuil fixe se recalent sur chaque image (lumière faible).
        col[s] = luminosité de la pièce relative à une case vide au même endroit."""
        lab = cv2.cvtColor(cv2.GaussianBlur(warped, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lg = np.log(cv2.GaussianBlur(gray, (3, 3), 0) + 8.0)
        gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3)
        lgrad = np.sqrt(gx * gx + gy * gy)

        pix, lab_med = {}, {}
        occ = np.zeros((64, 3))
        for s in chess.SQUARES:
            ys, xs = _square_slice(s)
            pix[s] = lab[ys, xs].reshape(-1, 3)
            lab_med[s] = np.median(pix[s], axis=0)
            g = lgrad[ys, xs]
            occ[s] = (float((g > EDGE_LOG_GRAD).mean()),
                      float(np.log(np.percentile(g, 90) + 0.02)),
                      float(np.log(g.mean() + 0.02)))

        base_fn = {grp: self._fit_empty_colour(lab_med, board, grp, exclude) for grp in (0, 1)}
        col = np.zeros(64)
        for s in chess.SQUARES:
            base = base_fn[square_group(s)](s)
            dev = np.linalg.norm(pix[s] - base, axis=1)
            k = max(1, len(dev) * 3 // 10)
            top = np.argpartition(dev, -k)[-k:]
            col[s] = float(pix[s][top, 0].mean() - base[0])
        return occ, col

    # --- Modèles ---------------------------------------------------------------------

    def _occupancy_probs(self, occ, labels, use):
        """P(case occupée), modèle vide/occupé par couleur de case."""
        p_occ = np.full(64, 0.5)
        for grp in (0, 1):
            sqs = [s for s in chess.SQUARES if square_group(s) == grp]
            emp = [occ[s] for s in sqs if use[s] and labels[s] == "E"]
            full = [occ[s] for s in sqs if use[s] and labels[s] != "E"]
            if len(emp) < 3:
                emp = [occ[s] for s in chess.SQUARES if use[s] and labels[s] == "E"]
            if len(full) < 3:
                full = [occ[s] for s in chess.SQUARES if use[s] and labels[s] != "E"]
            if len(emp) < 3 or len(full) < 3:
                continue
            me, se = _robust(emp, _OCC_FLOOR)
            mo, so = _robust(full, _OCC_FLOOR)
            for s in sqs:
                le, lo = _student_ll(occ[s], me, se), _student_ll(occ[s], mo, so)
                p_occ[s] = 1.0 / (1.0 + np.exp(np.clip(le - lo, -50, 50)))
        return p_occ

    def _colour_probs(self, col, labels, use):
        """P(blanche | occupée), modèle par couleur de case (repli : tous groupes confondus)."""
        p_white = np.full(64, 0.5)
        for grp in (0, 1):
            sqs = [s for s in chess.SQUARES if square_group(s) == grp]
            models = {}
            for c in ("W", "B"):
                rows = [col[s] for s in sqs if use[s] and labels[s] == c]
                if len(rows) >= 2:
                    models[c] = _robust(rows, _COL_FLOOR)
            if len(models) < 2:
                continue  # couleur indécidable pour ce groupe : 50/50 (ne départage rien)
            (mw, sw), (mb, sb) = models["W"], models["B"]
            for s in sqs:
                lw, lb = _student_ll(col[s], mw, sw), _student_ll(col[s], mb, sb)
                p_white[s] = 1.0 / (1.0 + np.exp(np.clip(lb - lw, -50, 50)))
        return p_white

    def _probs(self, warped, board, labels, use, exclude=()):
        occ, col = self._features(warped, board, exclude)
        p_occ = self._occupancy_probs(occ, labels, use)
        p_white = self._colour_probs(col, labels, use)
        probs = np.stack([1.0 - p_occ, p_occ * p_white, p_occ * (1.0 - p_white)], axis=1)
        probs = np.clip(probs, P_MIN, P_MAX)
        return probs / probs.sum(axis=1, keepdims=True)

    def read(self, warped, board, center_xy=None, warped_padded=None, config=None):
        """Probabilités (64, 3) pour E, W, B, indexées par case python-chess."""
        labels = {s: _label(board, s) for s in chess.SQUARES}
        idx = {c: i for i, c in enumerate(CLASSES)}
        use = {s: True for s in chess.SQUARES}
        probs = self._probs(warped, board, labels, use)

        # Seconde passe sans les cases qui contredisent nettement leur étiquette
        # (probablement celles d'un coup qui vient d'être joué)
        suspects = {s for s in chess.SQUARES if probs[s, idx[labels[s]]] < 0.15}
        if suspects:
            use = {s: s not in suspects for s in chess.SQUARES}
            probs = self._probs(warped, board, labels, use, exclude=suspects)

        # Intégration des pastilles de couleur (y compris celles débordant du quadrillage via warped_padded)
        pastille_src = warped_padded if warped_padded is not None else warped
        pastilles = detect_pastilles_by_square(pastille_src, center_xy=center_xy, config=config)
        self.last_pastilles = pastilles
        for sq, p in pastilles.items():
            if p.type in (PASTILLE_PAWN_B, PASTILLE_PIECE_B):
                # Pièce Noire (pion violet ou pièce bleue) : confirmation forte
                probs[sq, 0] = min(probs[sq, 0], 0.04)
                probs[sq, 1] = min(probs[sq, 1], 0.04)
                probs[sq, 2] = max(probs[sq, 2], 0.92)
            elif p.type in (PASTILLE_PAWN_W, PASTILLE_PIECE_W):
                # Pièce Blanche (pion rouge ou pièce verte) : confirmation forte
                probs[sq, 0] = min(probs[sq, 0], 0.04)
                probs[sq, 1] = max(probs[sq, 1], 0.92)
                probs[sq, 2] = min(probs[sq, 2], 0.04)
            total = probs[sq].sum()
            if total > 0:
                probs[sq] /= total

        # Pour les cases SANS pastille lorsque des pastilles sont actives :
        # Empêche qu'une ombre projetée (ex. lampe à côté ou grande pièce) ne soit classée
        # faussement comme une pièce noire. Une case sans pastille est très probablement vide.
        cfg = config or {}
        strict_pastilles = (cfg.get('piece_mode') == 'PASTILLES_STRICT') or (len(pastilles) >= 8)
        if strict_pastilles:
            for sq in chess.SQUARES:
                if sq not in pastilles:
                    if board.piece_at(sq) is None:
                        # Case attendue vide et sans pastille : confirmation vide forte
                        probs[sq, 0] = max(probs[sq, 0], 0.88)
                        probs[sq, 1] = min(probs[sq, 1], 0.06)
                        probs[sq, 2] = min(probs[sq, 2], 0.06)
                    else:
                        # Case qui avait une pièce mais n'a plus de pastille (pièce partie)
                        probs[sq, 0] = max(probs[sq, 0], 0.82)
                        probs[sq, 1] = min(probs[sq, 1], 0.09)
                        probs[sq, 2] = min(probs[sq, 2], 0.09)
                    probs[sq] /= probs[sq].sum()

        self.last_probs = probs
        return probs
