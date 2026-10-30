import os
import time
from pathlib import Path
import cv2
import chess

CLASS_NAMES = [
    "empty", "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK"
]

class DatasetCollector:
    """
    Découpe et sauvegarde automatiquement les 64 cases d'un échiquier
    dans les 13 dossiers correspondants (zéro labellisation manuelle).
    """
    def __init__(self, dataset_dir="dataset", log_fn=None):
        self.dataset_dir = Path(dataset_dir)
        self.log_fn = log_fn or print
        self._ensure_folders()

    def _ensure_folders(self):
        for c in CLASS_NAMES:
            (self.dataset_dir / c).mkdir(parents=True, exist_ok=True)

    def extract_64_patches(self, warped_512, pad=8, out_size=96):
        """
        Extrait les 64 vignettes avec une marge contextuelle (pad) pour capturer
        le haut des pièces même si la caméra est légèrement de biais.
        """
        if warped_512 is None:
            return {}

        bordered = cv2.copyMakeBorder(warped_512, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        patches = {}

        for sq in range(64):
            col = sq % 8
            row = 7 - (sq // 8)

            y1 = row * 64
            y2 = (row + 1) * 64 + 2 * pad
            x1 = col * 64
            x2 = (col + 1) * 64 + 2 * pad

            patch = bordered[y1:y2, x1:x2]
            patch_resized = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
            patches[sq] = patch_resized

        return patches

    def save_labeled_board(self, warped_512, board):
        """
        Sauvegarde les 64 cases de l'image redressée en les classant selon le plateau d'échecs.
        """
        if warped_512 is None or board is None:
            return False, "Image ou plateau manquant", 0

        patches = self.extract_64_patches(warped_512)
        saved_count = 0
        ts = int(time.time() * 1000)

        for sq, patch in patches.items():
            piece = board.piece_at(sq)
            if piece is None:
                label = "empty"
                # Sauvegarder seulement un échantillon des cases vides pour éviter le sur-déséquilibre
                if (sq % 3) != 0:
                    continue
            else:
                color_prefix = "w" if piece.color == chess.WHITE else "b"
                label = color_prefix + piece.symbol().upper()

            sq_name = chess.square_name(sq)
            filename = f"snap_{ts}_{sq_name}.jpg"
            target_path = self.dataset_dir / label / filename

            cv2.imwrite(str(target_path), patch)
            saved_count += 1

        return True, f"{saved_count} images sauvegardées dans {self.dataset_dir}", saved_count

    def get_dataset_stats(self):
        stats = {}
        total = 0
        for c in CLASS_NAMES:
            count = len(list((self.dataset_dir / c).glob("*.jpg")))
            stats[c] = count
            total += count
        return {
            "total_images": total,
            "classes": stats,
            "ready_for_training": total >= 64
        }
