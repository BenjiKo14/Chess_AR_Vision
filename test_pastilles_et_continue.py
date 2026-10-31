"""
Tests unitaires pour la détection des pastilles de couleur et la détection continue 5s.
"""

import unittest
import numpy as np
import cv2
import chess

from pastille_detector import (
    detect_pastille_blobs,
    detect_pastilles_by_square,
    PASTILLE_PAWN_B,
    PASTILLE_PIECE_B,
    PASTILLE_PAWN_W,
    PASTILLE_PIECE_W
)
from board_reader import BoardReader
from chess_vision import ChessVisionManager


class TestPastillesEtContinue(unittest.TestCase):
    def setUp(self):
        # Créer une image synthétique 512x512 avec les 4 types de pastilles
        self.img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Remplir un damier de fond grisâtre
        for r in range(8):
            for c in range(8):
                val = 220 if (r + c) % 2 == 0 else 60
                self.img[r*64:(r+1)*64, c*64:(c+1)*64] = (val, val, val)

        # Dessiner une pastille Violette (Pion Noir) au centre de la case (1, 1) -> sq b7
        cv2.circle(self.img, (1*64 + 32, 1*64 + 32), 7, (210, 50, 200), -1)

        # Dessiner une pastille Bleue (Pièce Noire) au centre de la case (0, 4) -> sq e8
        cv2.circle(self.img, (4*64 + 32, 0*64 + 32), 7, (240, 140, 30), -1)

        # Dessiner une pastille Rouge (Pion Blanc) au centre de la case (6, 3) -> sq d2
        cv2.circle(self.img, (3*64 + 32, 6*64 + 32), 7, (30, 30, 230), -1)

        # Dessiner une pastille Verte (Pièce Blanche) au centre de la case (7, 6) -> sq g1
        cv2.circle(self.img, (6*64 + 32, 7*64 + 32), 7, (40, 210, 40), -1)

    def test_pastille_blobs_detection(self):
        blobs = detect_pastille_blobs(self.img)
        self.assertGreaterEqual(len(blobs), 4)
        types_found = {b.type for b in blobs}
        self.assertIn(PASTILLE_PAWN_B, types_found)
        self.assertIn(PASTILLE_PIECE_B, types_found)
        self.assertIn(PASTILLE_PAWN_W, types_found)
        self.assertIn(PASTILLE_PIECE_W, types_found)

    def test_pastille_by_square_mapping(self):
        sq_map = detect_pastilles_by_square(self.img)
        # b7 = sq 49
        sq_b7 = chess.parse_square('b7')
        self.assertIn(sq_b7, sq_map)
        self.assertEqual(sq_map[sq_b7].type, PASTILLE_PAWN_B)

        # e8 = sq 60
        sq_e8 = chess.parse_square('e8')
        self.assertIn(sq_e8, sq_map)
        self.assertEqual(sq_map[sq_e8].type, PASTILLE_PIECE_B)

        # d2 = sq 11
        sq_d2 = chess.parse_square('d2')
        self.assertIn(sq_d2, sq_map)
        self.assertEqual(sq_map[sq_d2].type, PASTILLE_PAWN_W)

        # g1 = sq 6
        sq_g1 = chess.parse_square('g1')
        self.assertIn(sq_g1, sq_map)
        self.assertEqual(sq_map[sq_g1].type, PASTILLE_PIECE_W)

    def test_board_reader_integration(self):
        reader = BoardReader()
        board = chess.Board()
        probs = reader.read(self.img, board)

        sq_b7 = chess.parse_square('b7')
        sq_d2 = chess.parse_square('d2')

        # b7 (pastille violette = pion noir) -> probabilité noir > 0.90
        # CLASSES = ("E", "W", "B") -> index 2 = B
        self.assertGreater(probs[sq_b7, 2], 0.90)

        # d2 (pastille rouge = pion blanc) -> probabilité blanc > 0.90
        # index 1 = W
        self.assertGreater(probs[sq_d2, 1], 0.90)

    def test_continuous_detection_flag(self):
        vm = ChessVisionManager()
        vm.continuous_detection = False
        self.assertFalse(vm.continuous_detection)
        vm.continuous_detection = True
        status = vm.get_status()
        self.assertTrue(status.get('continuous_detection'))


if __name__ == '__main__':
    unittest.main()
