import json
from pathlib import Path
import cv2
import numpy as np
import chess
from dataset_collector import DatasetCollector, CLASS_NAMES

class OnnxPieceClassifier:
    """
    Inférence ultra-légère 100% CPU (< 50ms pour 64 cases)
    via le moteur cv2.dnn intégré nativement dans OpenCV.
    Aucune dépendance PyTorch requise sur le PC portable ou Raspberry Pi !
    """
    def __init__(self, model_path="models/chess_mobilenetv3.onnx", log_fn=None):
        self.model_path = Path(model_path)
        self.log_fn = log_fn or print
        self.net = None
        self.classes = CLASS_NAMES
        self.collector = DatasetCollector()
        self._load_model()

    def log(self, msg):
        try:
            self.log_fn(msg)
        except UnicodeEncodeError:
            try:
                safe_msg = str(msg).encode("ascii", errors="replace").decode("ascii")
                self.log_fn(safe_msg)
            except Exception:
                pass
        except Exception:
            pass

    def _load_model(self):
        if not self.model_path.exists():
            return False

        try:
            meta_path = self.model_path.parent / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.classes = data.get("classes", CLASS_NAMES)

            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.log("[OK] Modele ONNX charge via OpenCV DNN (100% CPU)")
            return True
        except Exception as e:
            self.log(f"[WARN] Erreur chargement ONNX: {e}")
            return False

    def is_ready(self):
        return self.net is not None

    def predict_board_fen(self, warped_512, active_turn=chess.WHITE):
        """
        Classifie les 64 cases d'un coup et retourne un FEN complet cohérent.
        """
        if not self.is_ready():
            return None, "Modèle ONNX non entraîné ou non chargé"

        patches_dict = self.collector.extract_64_patches(warped_512, pad=8, out_size=96)
        patches_list = [patches_dict[sq] for sq in range(64)]

        # Construction du batch 64x3x96x96
        blob = cv2.dnn.blobFromImages(
            patches_list,
            scalefactor=1.0 / 255.0,
            size=(96, 96),
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            swapRB=True,
            crop=False
        )
        # Ajuster écart-type
        blob[:, 0, :, :] /= 0.229
        blob[:, 1, :, :] /= 0.224
        blob[:, 2, :, :] /= 0.225

        self.net.setInput(blob)
        logits = self.net.forward()  # (64, 13)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Règle 1: Pions interdits sur rangées 1 et 8
        wp_idx = self.classes.index("wP") if "wP" in self.classes else -1
        bp_idx = self.classes.index("bP") if "bP" in self.classes else -1
        for sq in range(64):
            rank = chess.square_rank(sq)
            if rank == 0 or rank == 7:
                if wp_idx != -1:
                    probs[sq, wp_idx] = 0.0
                if bp_idx != -1:
                    probs[sq, bp_idx] = 0.0

        # Règle 2: Exactement 1 Roi Blanc et 1 Roi Noir (sur deux cases distinctes)
        wk_idx = self.classes.index("wK") if "wK" in self.classes else -1
        bk_idx = self.classes.index("bK") if "bK" in self.classes else -1
        if wk_idx != -1:
            best_wk_sq = int(np.argmax(probs[:, wk_idx]))
            probs[:, wk_idx] = 0.0
            probs[best_wk_sq, :] = 0.0
            probs[best_wk_sq, wk_idx] = 1.0

        if bk_idx != -1:
            bk_scores = probs[:, bk_idx].copy()
            if wk_idx != -1:
                bk_scores[best_wk_sq] = -1.0
            best_bk_sq = int(np.argmax(bk_scores))
            probs[:, bk_idx] = 0.0
            probs[best_bk_sq, :] = 0.0
            probs[best_bk_sq, bk_idx] = 1.0

        board = chess.Board(None)  # Plateau vide
        low_confidence_squares = []

        for sq in range(64):
            best_cls_idx = int(np.argmax(probs[sq]))
            conf = float(probs[sq, best_cls_idx])
            cls_name = self.classes[best_cls_idx]

            if conf < 0.65:
                low_confidence_squares.append({
                    "square": chess.square_name(sq),
                    "pred": cls_name,
                    "confidence": round(conf, 2)
                })

            if cls_name != "empty":
                color = chess.WHITE if cls_name.startswith("w") else chess.BLACK
                symbol = cls_name[1].upper()
                ptype = {
                    "P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP,
                    "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING
                }.get(symbol)
                if ptype:
                    board.set_piece_at(sq, chess.Piece(ptype, color))

        board.turn = active_turn
        fen = board.fen()
        return {
            "fen": fen,
            "board": board,
            "low_confidence_squares": low_confidence_squares,
            "avg_confidence": round(float(np.mean(np.max(probs, axis=1))), 3)
        }, None
