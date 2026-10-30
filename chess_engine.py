import os
import sys
import shutil
import random
import threading
import chess
import chess.engine

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

# Tables de valeurs positionnelles (Piece-Square Tables)
# Valeurs centrées sur le contrôle du centre et la sécurité
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

class ChessEngineManager:
    """
    Gère le bot d'échecs multi-niveaux.
    Supporte Stockfish UCI (si présent) avec fallback immédiat sur un moteur Minimax
    avec élagage Alpha-Beta et tables positionnelles, modulable de 600 à 2800 ELO.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or safe_log_print
        self.stockfish_path = None
        self.stockfish_available = False
        self.bot_elo = 1200
        self.game_mode = "HUMAN_VS_BOT"   # ou "HUMAN_VS_HUMAN"
        self.human_color = chess.WHITE     # ou chess.BLACK
        self._detect_stockfish()

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

    def _detect_stockfish(self, custom_path=None):
        candidates = []
        if custom_path:
            candidates.append(custom_path)

        env_path = os.environ.get("STOCKFISH_PATH")
        if env_path:
            candidates.append(env_path)

        local_dir = os.path.join(os.path.dirname(__file__), "engines")
        if os.path.exists(local_dir):
            for f in os.listdir(local_dir):
                if f.lower().startswith("stockfish"):
                    candidates.append(os.path.join(local_dir, f))

        which_sf = shutil.which("stockfish")
        if which_sf:
            candidates.append(which_sf)

        # Linux / Raspberry Pi
        candidates.extend(["/usr/games/stockfish", "/usr/bin/stockfish"])

        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK | os.R_OK):
                try:
                    engine = chess.engine.SimpleEngine.popen_uci(c)
                    engine.quit()
                    self.stockfish_path = c
                    self.stockfish_available = True
                    self.log(f"✅ Moteur Stockfish détecté : {c}")
                    return True
                except Exception:
                    pass

        self.stockfish_available = False
        self.log("ℹ️ Moteur Stockfish non trouvé -> Moteur Python Alpha-Beta actif par défaut (0% GPU, compatible Raspberry Pi)")
        return False

    def evaluate_board_static(self, board):
        """Évaluation statique matérielle + positionnelle."""
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue

            val = PIECE_VALUES.get(piece.piece_type, 0)
            pst_val = 0
            if piece.piece_type == chess.PAWN:
                pst_val = PAWN_TABLE[sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
            elif piece.piece_type == chess.KNIGHT:
                pst_val = KNIGHT_TABLE[sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
            elif piece.piece_type == chess.BISHOP:
                pst_val = BISHOP_TABLE[sq if piece.color == chess.WHITE else chess.square_mirror(sq)]

            total = val + pst_val
            score += total if piece.color == chess.WHITE else -total

        return score if board.turn == chess.WHITE else -score

    def _alphabeta(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board_static(board)

        # Tri basique : captures d'abord
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: 1 if board.is_capture(m) else 0, reverse=True)

        best_score = -999999
        for m in moves:
            board.push(m)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def compute_move(self, board, elo_target=None, time_limit=0.5):
        """Calcule le coup du Bot selon le niveau ELO ciblé."""
        target_elo = elo_target if elo_target is not None else self.bot_elo
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Option Stockfish si disponible et niveau > 1300
        if self.stockfish_available and target_elo > 1300:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                try:
                    skill = min(max(int((target_elo - 800) / 100), 0), 20)
                    engine.configure({"Skill Level": skill})
                except Exception:
                    pass

                result = engine.play(board, chess.engine.Limit(time=time_limit))
                engine.quit()
                if result.move:
                    return {
                        "move": result.move,
                        "uci": result.move.uci(),
                        "san": board.san(result.move),
                        "engine": "Stockfish UCI",
                        "elo": target_elo
                    }
            except Exception as e:
                self.log(f"Erreur Stockfish, fallback Python: {e}")

        # Moteur Python Alpha-Beta avec ajustement de profondeur et bruit ELO
        if target_elo <= 900:
            depth = 1
            noise_sigma = 120
        elif target_elo <= 1200:
            depth = 2
            noise_sigma = 60
        elif target_elo <= 1500:
            depth = 2
            noise_sigma = 20
        else:
            depth = 3
            noise_sigma = 0

        scored_moves = []
        for m in legal_moves:
            board.push(m)
            base_score = -self._alphabeta(board, depth - 1, -999999, 999999)
            board.pop()

            noise = random.gauss(0, noise_sigma) if noise_sigma > 0 else 0
            scored_moves.append((m, base_score + noise))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        best_move = scored_moves[0][0]

        return {
            "move": best_move,
            "uci": best_move.uci(),
            "san": board.san(best_move),
            "engine": "Python Alpha-Beta PST",
            "elo": target_elo
        }

    def get_hint(self, board):
        """Donne un conseil / indice au joueur humain."""
        target = 2200 if self.stockfish_available else 1200
        res = self.compute_move(board, elo_target=target, time_limit=0.4)
        return res
