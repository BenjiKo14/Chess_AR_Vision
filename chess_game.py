import sys
import time
import threading
import chess

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

class ChessGameManager:
    """
    Gère la machine à états de la partie d'échecs,
    la détection des coups légaux par vision différentielle,
    l'historique et la synchronisation avec le bot AR.
    """
    def __init__(self, vision_manager, engine_manager, log_fn=None):
        self.vision = vision_manager
        self.engine = engine_manager
        self.log_fn = log_fn or safe_log_print
        self._lock = threading.RLock()

        self.board = chess.Board()
        # États : "IDLE", "SETUP_CHECK", "WAIT_HUMAN_MOVE", "WAIT_BOT_THINKING", "WAIT_BOT_EXECUTION", "GAME_OVER"
        self.state = "IDLE"

        self.reference_warped = None
        self.last_move = None
        self.move_history = []  # Liste de dict: {num, uci, san, fen}
        self.pending_bot_move = None

        self.consecutive_match_count = 0
        self.last_candidate_move = None
        self.min_change_threshold = 13.0

        # Hook d'information pour le pont robot
        self.on_move_played_hook = None

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

    def start_new_game(self, fen=None):
        """Démarre une nouvelle partie."""
        with self._lock:
            self.board = chess.Board(fen) if fen else chess.Board()
            self.move_history.clear()
            self.last_move = None
            self.pending_bot_move = None
            self.vision.ar_arrow = None
            self.vision.highlight_squares.clear()
            self.vision.setup_mode = True
            self.state = "SETUP_CHECK"
            self.log("♟️ Mode vérification des pièces au départ activé")
            return self.get_status()

    def confirm_setup(self):
        """Valide le plateau initial et capture l'image de référence."""
        with self._lock:
            cur = self.vision.get_latest_warped_frame()
            if cur is None:
                return False, "Aucune image reçue de la caméra"

            self.reference_warped = cur.copy()
            self.vision.setup_mode = False

            # Déterminer qui commence
            if self.engine.game_mode == "HUMAN_VS_BOT" and self.board.turn != self.engine.human_color:
                self.state = "WAIT_BOT_THINKING"
                threading.Thread(target=self._bot_turn_thread, daemon=True).start()
            else:
                self.state = "WAIT_HUMAN_MOVE"

            self.log("✅ Plateau validé. La partie commence !")
            return True, "Partie commencée"

    def resync_reference(self):
        """Recalibre l'image de référence sur l'état visuel actuel (J'adoube / lumière)."""
        with self._lock:
            cur = self.vision.get_latest_warped_frame()
            if cur is not None:
                self.reference_warped = cur.copy()
                self.log("📸 Image de référence resynchronisée.")
                return True
        return False

    def update_vision_tick(self):
        """Appelé périodiquement pour détecter les coups physiques quand l'image est stable."""
        with self._lock:
            if self.state not in ("WAIT_HUMAN_MOVE", "WAIT_BOT_EXECUTION"):
                return
            if not self.vision.is_board_stable or self.reference_warped is None:
                return

            cur_warped = self.vision.get_latest_warped_frame()
            if cur_warped is None:
                return

            changes = self.vision.compute_all_square_changes(self.reference_warped, cur_warped)

            # Info de débogage top 2 cases
            sorted_sq = sorted(changes.items(), key=lambda x: x[1], reverse=True)
            if sorted_sq and sorted_sq[0][1] > 6.0:
                s1_name = chess.square_name(sorted_sq[0][0])
                s2_name = chess.square_name(sorted_sq[1][0])
                self.vision.top_diff_info = f"Top diff: {s1_name}({sorted_sq[0][1]}), {s2_name}({sorted_sq[1][1]})"
            else:
                self.vision.top_diff_info = ""

            if self.state == "WAIT_HUMAN_MOVE":
                self._check_human_move(changes, cur_warped)
            elif self.state == "WAIT_BOT_EXECUTION":
                self._check_bot_execution(changes, cur_warped)

    def _check_human_move(self, changes, cur_warped):
        # Vérifier si au moins deux cases dépassent le seuil
        significant_squares = [sq for sq, score in changes.items() if score >= self.min_change_threshold]
        if len(significant_squares) < 2:
            self.consecutive_match_count = 0
            self.last_candidate_move = None
            return

        best_move = None
        best_score = -999.0

        for m in self.board.legal_moves:
            expected = {m.from_square, m.to_square}
            if self.board.is_castling(m):
                # Ajout de la Tour
                rank = chess.square_rank(m.from_square)
                if chess.square_file(m.to_square) == 6:  # Petit roque
                    expected.update({chess.square(7, rank), chess.square(5, rank)})
                else:  # Grand roque
                    expected.update({chess.square(0, rank), chess.square(3, rank)})
            elif self.board.is_en_passant(m):
                # Pion capturé
                cap_sq = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
                expected.add(cap_sq)

            # Vérifier que toutes les cases attendues ont un score de changement suffisant
            if all(changes.get(sq, 0.0) >= self.min_change_threshold for sq in expected):
                pos_score = sum(changes.get(sq, 0.0) for sq in expected) / len(expected)
                # Bruit = pénalité des cases non attendues qui auraient bougé
                other_scores = [changes.get(sq, 0.0) for sq in significant_squares if sq not in expected]
                noise = max(other_scores) if other_scores else 0.0
                score = pos_score - 0.7 * noise

                if score > best_score:
                    best_score = score
                    best_move = m

        if best_move is not None:
            if self.last_candidate_move == best_move:
                self.consecutive_match_count += 1
            else:
                self.last_candidate_move = best_move
                self.consecutive_match_count = 1

            # Confirmer après 2 vérifications stables consécutives (~0.4s)
            if self.consecutive_match_count >= 2:
                self._apply_move(best_move, cur_warped, player="human")
                self.consecutive_match_count = 0
                self.last_candidate_move = None

    def _check_bot_execution(self, changes, cur_warped):
        """Vérifie que la pièce du Bot a bien été déplacée physiquement."""
        if not self.pending_bot_move:
            return

        m = self.pending_bot_move["move"]
        f_sq = m.from_square
        t_sq = m.to_square

        score_f = changes.get(f_sq, 0.0)
        score_t = changes.get(t_sq, 0.0)

        if score_f >= self.min_change_threshold and score_t >= self.min_change_threshold:
            self.consecutive_match_count += 1
            if self.consecutive_match_count >= 2:
                self.log(f"🤖 Coup du bot {m.uci()} exécuté physiquement avec succès !")
                self._apply_move(m, cur_warped, player="bot")
                self.pending_bot_move = None
                self.vision.ar_arrow = None
                self.consecutive_match_count = 0

    def _apply_move(self, move, cur_warped, player="human"):
        san = self.board.san(move)
        uci = move.uci()
        self.board.push(move)
        self.reference_warped = cur_warped.copy()
        self.last_move = uci

        # Surbrillance AR du dernier coup
        self.vision.highlight_squares = {
            move.from_square: {"color": (255, 180, 0), "alpha": 0.35},
            move.to_square: {"color": (0, 220, 100), "alpha": 0.35}
        }

        # Échec ?
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                self.vision.highlight_squares[king_sq] = {"color": (30, 30, 240), "alpha": 0.6}

        entry = {
            "num": len(self.board.move_stack),
            "uci": uci,
            "san": san,
            "player": player,
            "fen": self.board.fen()
        }
        self.move_history.append(entry)
        self.log(f"♟️ Coup validé: {san} ({uci}) [{player}]")

        # Notification pour pont robot
        if self.on_move_played_hook:
            try:
                self.on_move_played_hook(entry)
            except Exception:
                pass

        if self.board.is_game_over():
            self.state = "GAME_OVER"
            self.log(f"🏁 Fin de partie : {self.board.result()}")
            return

        # Tour suivant
        if self.engine.game_mode == "HUMAN_VS_BOT" and player == "human":
            self.state = "WAIT_BOT_THINKING"
            threading.Thread(target=self._bot_turn_thread, daemon=True).start()
        else:
            self.state = "WAIT_HUMAN_MOVE"

    def _bot_turn_thread(self):
        """Thread asynchrone pour calculer le coup du Bot sans geler la vidéo."""
        time.sleep(0.3)
        res = self.engine.compute_move(self.board)
        if not res:
            return

        with self._lock:
            move = res["move"]
            self.pending_bot_move = {
                "move": move,
                "uci": res["uci"],
                "san": res["san"],
                "is_capture": self.board.is_capture(move),
                "is_castling": self.board.is_castling(move)
            }
            # Afficher la flèche AR
            self.vision.ar_arrow = {
                "from_sq": move.from_square,
                "to_sq": move.to_square,
                "is_capture": self.board.is_capture(move),
                "color": (0, 200, 255) if not self.board.is_capture(move) else (40, 50, 240),
                "label": f"Bot: {res['san']}"
            }
            self.state = "WAIT_BOT_EXECUTION"
            self.log(f"🤖 Bot veut jouer : {res['san']} ({res['uci']}) - Déplacez la pièce sur le plateau")

    def play_manual_move(self, uci_str):
        """Joue un coup manuellement via l'interface Web."""
        with self._lock:
            try:
                move = chess.Move.from_uci(uci_str)
                if move in self.board.legal_moves:
                    cur = self.vision.get_latest_warped_frame()
                    if cur is None:
                        cur = self.reference_warped
                    self._apply_move(move, cur, player="manual")
                    return True, "Coup joué"
                return False, "Coup illégal"
            except Exception as e:
                return False, str(e)

    def undo_last_move(self):
        """Annule le dernier demi-coup."""
        with self._lock:
            if self.board.move_stack:
                self.board.pop()
                if self.move_history:
                    self.move_history.pop()
                cur = self.vision.get_latest_warped_frame()
                if cur is not None:
                    self.reference_warped = cur.copy()
                self.last_move = self.move_history[-1]["uci"] if self.move_history else None
                self.vision.ar_arrow = None
                self.vision.highlight_squares.clear()
                self.pending_bot_move = None
                self.state = "WAIT_HUMAN_MOVE"
                return True
        return False

    def get_status(self):
        with self._lock:
            return {
                "state": self.state,
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "is_check": self.board.is_check(),
                "is_game_over": self.board.is_game_over(),
                "result": self.board.result() if self.board.is_game_over() else None,
                "last_move": self.last_move,
                "pending_bot_move": {
                    "uci": self.pending_bot_move["uci"],
                    "san": self.pending_bot_move["san"],
                    "is_capture": self.pending_bot_move["is_capture"]
                } if self.pending_bot_move else None,
                "history": self.move_history[-10:]
            }
