import os
import sys
import shutil
import tempfile
import unittest
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import chess
import torch

from dataset_collector import DatasetCollector, CLASS_NAMES
from onnx_classifier import OnnxPieceClassifier


class TestSlice5(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_collector_extract_patches(self):
        """Vérifie le découpage de 64 vignettes 96x96 depuis une image 512x512."""
        collector = DatasetCollector(dataset_dir=os.path.join(self.temp_dir, "dataset"))
        synthetic_img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Dessiner un motif pour vérifier que les patches contiennent des pixels
        cv2.circle(synthetic_img, (256, 256), 100, (255, 255, 255), -1)

        patches = collector.extract_64_patches(synthetic_img, pad=8, out_size=96)
        
        self.assertEqual(len(patches), 64, "Doit extraire exactement 64 vignettes")
        for sq in range(64):
            self.assertIn(sq, patches, f"La case {sq} doit être présente")
            patch = patches[sq]
            self.assertEqual(patch.shape, (96, 96, 3), f"Le patch {sq} doit mesurer 96x96x3")

    def test_dataset_collector_save_labeled_board(self):
        """Vérifie la sauvegarde labellisée selon le plateau chess.Board."""
        dataset_path = os.path.join(self.temp_dir, "dataset")
        collector = DatasetCollector(dataset_dir=dataset_path)
        synthetic_img = np.full((512, 512, 3), 128, dtype=np.uint8)
        board = chess.Board()  # Position de départ standard

        success, msg, saved_count = collector.save_labeled_board(synthetic_img, board)
        self.assertTrue(success)
        self.assertGreater(saved_count, 0)

        stats = collector.get_dataset_stats()
        self.assertEqual(stats["total_images"], saved_count)
        self.assertIn("classes", stats)
        # Vérifier que les dossiers de pièces existent et contiennent des images
        self.assertGreater(stats["classes"]["wP"], 0)
        self.assertGreater(stats["classes"]["bP"], 0)

    def test_onnx_classifier_absence_model(self):
        """Vérifie la gestion sans crash lors de l'absence de fichier ONNX."""
        non_existent_path = os.path.join(self.temp_dir, "missing_model.onnx")
        classifier = OnnxPieceClassifier(model_path=non_existent_path)

        self.assertFalse(classifier.is_ready())
        synthetic_img = np.zeros((512, 512, 3), dtype=np.uint8)
        res, err = classifier.predict_board_fen(synthetic_img)
        self.assertIsNone(res)
        self.assertIn("non entraîné ou non chargé", err)

    def test_onnx_classifier_corrupt_model(self):
        """Vérifie la gestion propre sans crash si le fichier ONNX est corrompu ou invalide."""
        corrupt_path = os.path.join(self.temp_dir, "corrupt.onnx")
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("NOT_A_VALID_ONNX_MODEL")

        classifier = OnnxPieceClassifier(model_path=corrupt_path)
        self.assertFalse(classifier.is_ready())
        res, err = classifier.predict_board_fen(np.zeros((512, 512, 3), dtype=np.uint8))
        self.assertIsNone(res)
        self.assertIsNotNone(err)

    def test_onnx_classifier_valid_model(self):
        """Vérifie l'inférence avec un modèle ONNX valide."""
        onnx_path = os.path.join(self.temp_dir, "dummy_model.onnx")

        # Modèle ultra-léger exporté en ONNX pour le test
        dummy_net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 13)
        )
        dummy_net.eval()
        dummy_input = torch.randn(64, 3, 96, 96)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                dummy_net,
                dummy_input,
                onnx_path,
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )

        classifier = OnnxPieceClassifier(model_path=onnx_path)
        self.assertTrue(classifier.is_ready())

        synthetic_img = np.zeros((512, 512, 3), dtype=np.uint8)
        res, err = classifier.predict_board_fen(synthetic_img, active_turn=chess.WHITE)

        self.assertIsNone(err)
        self.assertIsNotNone(res)
        self.assertIn("fen", res)
        self.assertIn("board", res)
        self.assertIn("avg_confidence", res)
        self.assertIsInstance(res["board"], chess.Board)
        self.assertEqual(res["board"].turn, chess.WHITE)

        # Vérification des règles échecs appliquées
        board = res["board"]
        # Exactement 1 Roi Blanc et 1 Roi Noir
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))
        self.assertEqual(white_kings, 1, "Doit avoir exactement 1 Roi Blanc")
        self.assertEqual(black_kings, 1, "Doit avoir exactement 1 Roi Noir")

        # Aucun pion sur rangées 1 et 8
        for sq in range(64):
            rank = chess.square_rank(sq)
            if rank in (0, 7):
                piece = board.piece_at(sq)
                if piece is not None:
                    self.assertNotEqual(piece.piece_type, chess.PAWN, f"Pion interdit sur la rangée {rank+1}")

    def test_train_piece_classifier_help(self):
        """Vérifie que train_piece_classifier.py s'exécute avec --help."""
        script_path = Path(__file__).parent / "train_piece_classifier.py"
        res = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--dataset", res.stdout)
        self.assertIn("--epochs", res.stdout)


if __name__ == "__main__":
    unittest.main()
