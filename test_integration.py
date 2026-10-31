import os
import sys
import unittest
import numpy as np
import cv2
import chess

from chess_vision import ChessVisionManager
from chess_engine import ChessEngineManager
from chess_game import ChessGameManager
from robot_bridge import RobotChessBridge


class TestIntegrationSlices1To4(unittest.TestCase):
    """
    Test d'intégration validant le fonctionnement conjoint des Slices 1 à 4 :
    - Slice 1 : ChessVisionManager (Homographie, projection AR, calibration)
    - Slice 2 : ChessGameManager (Différence visuelle, détection coups, machine d'état)
    - Slice 3 : ChessEngineManager (Bot Minimax Alpha-Beta, conseil AR)
    - Slice 4 : RobotChessBridge (Traduction des coups en actions physiques pour robot SO-101)
    """

    def setUp(self):
        # Initialisation avec les loggers par défaut pour vérifier l'absence d'UnicodeEncodeError
        self.vision = ChessVisionManager(calibration_file="test_board_calib.json")
        self.engine = ChessEngineManager()
        self.game = ChessGameManager(self.vision, self.engine, save_path="test_game_state_tmp.json")
        self.robot = RobotChessBridge(square_size_mm=45.0)

    def tearDown(self):
        import os
        if os.path.exists("test_game_state_tmp.json"):
            os.remove("test_game_state_tmp.json")
        calib_file = "test_board_calib.json"
        if os.path.exists(calib_file):
            try:
                os.remove(calib_file)
            except OSError:
                pass

    def test_unicode_safe_logging(self):
        """Vérifie que les méthodes log() ne provoquent aucune exception même avec émojis."""
        test_messages = [
            "♟️ Mode vérification des pièces au départ activé",
            "✅ Calibration plateau chargée depuis board_calibration.json",
            "🤖 Moteur Minimax initialisé",
            "⚠️ Erreur chargement calibration",
            "💾 Calibration sauvegardée.",
            "❌ Erreur sauvegarde calibration",
            "ℹ️ Moteur Stockfish non trouvé -> Moteur Python actif"
        ]
        for msg in test_messages:
            self.vision.log(msg)
            self.engine.log(msg)
            self.game.log(msg)
            self.robot.log(msg)

    def test_vision_perspective_and_ar(self):
        """Valide le calcul d'homographie, le redressement et la projection AR."""
        # Image de synthèse 640x480
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Définir 4 coins
        corners = [[0.1, 0.1], [0.9, 0.1], [0.85, 0.9], [0.15, 0.9]]
        self.vision.set_corners(corners)
        self.assertTrue(self.vision.is_calibrated)

        # Calculer homographie
        ok = self.vision.update_homography((480, 640))
        self.assertTrue(ok)
        self.assertIsNotNone(self.vision.H)
        self.assertIsNotNone(self.vision.H_inv)

        # Redressement 512x512
        warped = self.vision.warp_frame(raw_frame)
        self.assertEqual(warped.shape, (512, 512, 3))

        # Projection de points redressés vers caméra
        cam_pts = self.vision.warped_points_to_camera([(256.0, 256.0)])
        self.assertEqual(len(cam_pts), 1)
        self.assertTrue(0 <= cam_pts[0][0] <= 640)
        self.assertTrue(0 <= cam_pts[0][1] <= 480)

    def test_engine_minimax_alphabeta(self):
        """Valide que le moteur Python Alpha-Beta génère des coups légaux."""
        board = chess.Board()
        move_info = self.engine.compute_move(board, elo_target=1200)
        self.assertIsNotNone(move_info)
        self.assertIn("move", move_info)
        self.assertIn("san", move_info)
        self.assertIn("uci", move_info)
        self.assertIn(move_info["move"], board.legal_moves)

        # Test demande de conseil / hint
        hint = self.engine.get_hint(board)
        self.assertIsNotNone(hint)
        self.assertIn(hint["move"], board.legal_moves)

    def test_game_flow_and_states(self):
        """Valide la gestion de l'échiquier et la transition d'états."""
        # Début de partie
        status = self.game.start_new_game()
        self.assertEqual(self.game.state, "SETUP_CHECK")
        self.assertTrue(self.vision.setup_mode)

        # Simuler une frame reçue par la vision
        synth_frame = np.full((512, 512, 3), 100, dtype=np.uint8)
        self.vision._latest_warped_frame = synth_frame

        # Confirmation du setup
        ok, msg = self.game.confirm_setup()
        self.assertTrue(ok)
        self.assertIn(self.game.state, ["WAIT_HUMAN_MOVE", "WAIT_BOT_THINKING"])

    def test_robot_bridge_actions(self):
        """Valide la décomposition des coups d'échecs en séquences robotiques atomiques."""
        board = chess.Board()
        self.vision.update_homography((480, 640))

        # 1. Coup simple (e2-e4)
        move_simple = chess.Move.from_uci("e2e4")
        plan_simple = self.robot.build_robot_action_plan(board, move_simple, self.vision)
        self.assertEqual(plan_simple["move_uci"], "e2e4")
        self.assertEqual(plan_simple["step_count"], 2)  # MOVE_PIECE + RETURN_HOME
        self.assertEqual(plan_simple["actions"][0]["action"], "MOVE_PIECE")
        self.assertEqual(plan_simple["actions"][0]["piece"], "pawn")
        self.assertEqual(plan_simple["actions"][1]["action"], "RETURN_HOME")

        # 2. Prise (ex: e4xd5)
        board_cap = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        move_cap = chess.Move.from_uci("e4d5")
        plan_cap = self.robot.build_robot_action_plan(board_cap, move_cap, self.vision)
        self.assertEqual(plan_cap["step_count"], 3)  # REMOVE_CAPTURED + MOVE_PIECE + RETURN_HOME
        self.assertEqual(plan_cap["actions"][0]["action"], "REMOVE_CAPTURED")
        self.assertEqual(plan_cap["actions"][1]["action"], "MOVE_PIECE")
        self.assertEqual(plan_cap["actions"][2]["action"], "RETURN_HOME")

        # 3. Roque (e1g1)
        board_castle = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        move_castle = chess.Move.from_uci("e1g1")
        plan_castle = self.robot.build_robot_action_plan(board_castle, move_castle, self.vision)
        self.assertEqual(plan_castle["step_count"], 3)  # MOVE_PIECE (Roi) + MOVE_ROOK_CASTLING + RETURN_HOME
        self.assertEqual(plan_castle["actions"][0]["action"], "MOVE_PIECE")
        self.assertEqual(plan_castle["actions"][1]["action"], "MOVE_ROOK_CASTLING")
        self.assertEqual(plan_castle["actions"][2]["action"], "RETURN_HOME")

    def test_full_pipeline_interaction(self):
        """Valide l'orchestration complète : Vision -> Game -> Engine -> Robot."""
        # 1. Vision et calibration
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.vision.set_corners([[0.1, 0.1], [0.9, 0.1], [0.85, 0.9], [0.15, 0.9]])
        self.vision.process_frame(raw_frame)

        # 2. Game manager démarre
        self.game.start_new_game()
        ok, _ = self.game.confirm_setup()
        self.assertTrue(ok)

        # 3. Engine calcule un coup
        calc = self.engine.compute_move(self.game.board, elo_target=1000)
        bot_move = calc["move"]
        self.assertIn(bot_move, self.game.board.legal_moves)

        # 4. Robot génère le plan d'action
        plan = self.robot.build_robot_action_plan(self.game.board, bot_move, self.vision)
        self.assertGreater(len(plan["actions"]), 1)
        self.assertEqual(plan["actions"][-1]["action"], "RETURN_HOME")

        # 5. Application du coup sur le plateau logique
        self.game.board.push(bot_move)
        self.assertEqual(len(self.game.board.move_stack), 1)


if __name__ == "__main__":
    unittest.main()
