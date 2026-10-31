import sys
import time
import json
import os
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
    def __init__(self, vision_manager, engine_manager, log_fn=None, save_path="game_state.json"):
        self.vision = vision_manager
        self.engine = engine_manager
        self.log_fn = log_fn or safe_log_print
        self._lock = threading.RLock()
        self.save_path = save_path

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
        # Incrémenté à chaque nouvelle partie / annulation : un calcul du bot
        # lancé avant devient obsolète et ne doit plus rien afficher.
        self._generation = 0
        # Texte des coups possibles quand la caméra ne peut pas trancher (sinon None)
        self.ambiguity = None

        # Hook d'information pour le pont robot
        self.on_move_played_hook = None

        # Charger la partie précédente si elle existe
        self._load_game()

    def _save_game(self):
        """Sauvegarde l'état actuel de la partie dans un fichier JSON pour reprise automatique."""
        try:
            data = {
                "fen": self.board.fen(),
                "state": self.state,
                "last_move": self.last_move,
                "move_history": self.move_history,
                "engine_mode": getattr(self.engine, "game_mode", "HUMAN_VS_BOT"),
                "human_color": getattr(self.engine, "human_color", 1),
                "bot_elo": getattr(self.engine, "bot_elo", 1200),
                "saved_at": time.time()
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"⚠️ Erreur sauvegarde partie ({self.save_path}): {e}")

    def _load_game(self):
        """Restaure la partie précédente depuis le fichier JSON si existant."""
        if not os.path.exists(self.save_path):
            return False
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fen = data.get("fen")
            if fen:
                self.board = chess.Board(fen)
                self.move_history = data.get("move_history", [])
                self.last_move = data.get("last_move")
                saved_state = data.get("state", "WAIT_HUMAN_MOVE")
                # Si la partie était en réflexion bot ou en cours, reprendre en attente du joueur ou coup
                if saved_state in ("WAIT_BOT_THINKING", "WAIT_BOT_EXECUTION"):
                    self.state = "WAIT_HUMAN_MOVE"
                else:
                    self.state = saved_state

                # Surbrillance du dernier coup si existant
                if self.last_move and len(self.last_move) >= 4:
                    try:
                        m = chess.Move.from_uci(self.last_move)
                        self.vision.highlight_squares = {
                            m.from_square: {"color": (255, 180, 0), "alpha": 0.35},
                            m.to_square: {"color": (0, 220, 100), "alpha": 0.35}
                        }
                    except Exception:
                        pass

                self.log(f"💾 Partie précédente restaurée : {len(self.move_history)} demi-coups (FEN: {fen[:25]}...)")
                return True
        except Exception as e:
            self.log(f"⚠️ Impossible de charger la partie précédente: {e}")
        return False

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
            self.reference_warped = None
            self.consecutive_match_count = 0
            self.last_candidate_move = None
            self._generation += 1
            self.ambiguity = None
            self.vision.ar_arrow = None
            self.vision.highlight_squares.clear()
            self.vision.setup_mode = True
            self.state = "SETUP_CHECK"
            self._save_game()
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
                self._start_bot_turn()
            else:
                self.state = "WAIT_HUMAN_MOVE"

            self.log("✅ Plateau validé. La partie commence !")
            return True, "Partie commencée"

    _STATUS_MESSAGES = [
        (chess.STATUS_NO_WHITE_KING, "il manque le roi blanc"),
        (chess.STATUS_NO_BLACK_KING, "il manque le roi noir"),
        (chess.STATUS_TOO_MANY_KINGS, "il y a trop de rois"),
        (chess.STATUS_OPPOSITE_CHECK, "le camp qui n'a pas le trait est en échec (vérifiez le trait)"),
        (chess.STATUS_PAWNS_ON_BACKRANK, "un pion est sur la première ou la dernière rangée"),
        (chess.STATUS_TOO_MANY_WHITE_PAWNS, "trop de pions blancs"),
        (chess.STATUS_TOO_MANY_BLACK_PAWNS, "trop de pions noirs"),
        (chess.STATUS_TOO_MANY_WHITE_PIECES, "trop de pièces blanches"),
        (chess.STATUS_TOO_MANY_BLACK_PIECES, "trop de pièces noires"),
    ]

    def set_position(self, fen):
        """Remplace la position connue par celle corrigée à la main par le joueur,
        puis reprend la partie depuis cette position (référence caméra = image actuelle)."""
        try:
            board = chess.Board(fen)
        except ValueError:
            return False, "Position illisible (FEN invalide)"

        with self._lock:
            # Le roque ne peut que se perdre : on garde les droits de l'ancienne
            # position encore cohérents avec le placement des rois et des tours.
            board.castling_rights = board.clean_castling_rights() & self.board.castling_rights
            board.ep_square = None

            status = board.status()
            if status != chess.STATUS_VALID:
                reasons = [msg for flag, msg in self._STATUS_MESSAGES if status & flag]
                return False, "Position impossible : " + (", ".join(reasons) or "position non valide")

            self.board = board
            self._generation += 1
            self.pending_bot_move = None
            self.ambiguity = None
            self.consecutive_match_count = 0
            self.last_candidate_move = None
            self.last_move = None
            self.vision.ar_arrow = None
            self.vision.highlight_squares.clear()
            self.vision.setup_mode = False
            self.move_history.append({
                "num": "✏️", "uci": "", "san": "Position corrigée",
                "player": "manuel", "fen": board.fen()
            })

            cur = self.vision.get_latest_warped_frame()
            if cur is not None:
                self.reference_warped = cur.copy()
                msg = "Position corrigée"
            else:
                msg = "Position corrigée, mais aucune image caméra : cliquez sur « Resync » pour réactiver la détection"
            self.log(f"✏️ {msg} : {board.fen()}")

            if board.is_game_over():
                self.state = "GAME_OVER"
                self.log(f"🏁 Fin de partie : {board.result()}")
            elif self.engine.game_mode == "HUMAN_VS_BOT" and board.turn != self.engine.human_color:
                self._start_bot_turn()
            else:
                self.state = "WAIT_HUMAN_MOVE"

            # Sauvegarde automatique de la position corrigée
            self._save_game()
            return True, msg

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

            self._detect_position_change(changes, cur_warped)

    def _expected_changed_squares(self, line):
        """Cases dont le contenu diffère entre la position connue et la position
        obtenue après la suite de coups `line`. Couvre roque, prise en passant,
        promotion et reprise sur la même case sans cas particulier."""
        after = self.board.copy(stack=False)
        for m in line:
            after.push(m)
        return {sq for sq in chess.SQUARES if self.board.piece_at(sq) != after.piece_at(sq)}

    @staticmethod
    def _camera_moves(board):
        """Coups légaux détectables à la caméra : la pièce de promotion ne se voit pas
        dans la différence d'image, on suppose la dame (sous-promotion : « Corriger la position »)."""
        return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]

    def _candidate_lines(self):
        """Suites de coups pouvant expliquer la différence entre l'image de référence et l'image actuelle."""
        if self.state == "WAIT_HUMAN_MOVE":
            return [[m] for m in self._camera_moves(self.board)]
        if self.state == "WAIT_BOT_EXECUTION" and self.pending_bot_move:
            bot = self.pending_bot_move["move"]
            lines = [[bot]]
            # Le joueur a pu enchaîner son coup juste après avoir déplacé la pièce
            # du bot, sans que le plateau redevienne stable entre les deux.
            if self.engine.game_mode == "HUMAN_VS_BOT":
                after = self.board.copy(stack=False)
                after.push(bot)
                if not after.is_game_over():
                    lines.extend([bot, h] for h in self._camera_moves(after))
            return lines
        return []

    def _score_line(self, line, changes, significant):
        """Score de correspondance entre une suite de coups et les changements observés.
        None si une case qui aurait dû changer n'a pas assez changé."""
        expected = self._expected_changed_squares(line)
        if not expected:
            return None
        if any(changes.get(sq, 0.0) < self.min_change_threshold for sq in expected):
            return None
        pos_score = sum(changes.get(sq, 0.0) for sq in expected) / len(expected)
        # Pénalité : cases qui ont changé sans être expliquées par la suite de coups
        unexplained = [changes.get(sq, 0.0) for sq in significant if sq not in expected]
        noise = max(unexplained) if unexplained else 0.0
        return pos_score - 0.7 * noise

    def _detect_position_change(self, changes, cur_warped):
        """Compare la position vue par la caméra à la position connue et cherche la
        suite de coups (1 ou 2 demi-coups) qui explique le mieux la différence."""
        significant = [sq for sq, score in changes.items() if score >= self.min_change_threshold]
        if len(significant) < 2:
            self.consecutive_match_count = 0
            self.last_candidate_move = None
            return

        scored = []
        for line in self._candidate_lines():
            score = self._score_line(line, changes, significant)
            if score is not None:
                scored.append((score, line))

        if not scored:
            self.consecutive_match_count = 0
            self.last_candidate_move = None
            return

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line = scored[0]

        # Deux suites différentes qui changent les mêmes cases (ex. reprendre sur la
        # case d'arrivée du bot ou aller sur sa case de départ) : la caméra ne peut
        # pas trancher. On ne devine pas, on demande une saisie manuelle.
        best_squares = self._expected_changed_squares(best_line)
        rivals = [l for _, l in scored[1:] if self._expected_changed_squares(l) == best_squares]
        if rivals:
            options = sorted(" puis ".join(m.uci() for m in l) for l in [best_line] + rivals)
            key = ("ambiguous",) + tuple(options)
            if self.last_candidate_move != key:  # ne logguer qu'une fois
                self.last_candidate_move = key
                self.ambiguity = " ou ".join(options)
                self.log(f"❓ Position ambiguë ({' ou '.join(options)}) : placez la bonne position via « ✏️ Corriger la position ».")
            self.consecutive_match_count = 0
            return

        self.ambiguity = None
        key = tuple(m.uci() for m in best_line)
        if self.last_candidate_move == key:
            self.consecutive_match_count += 1
        else:
            self.last_candidate_move = key
            self.consecutive_match_count = 1

        # Confirmer après 2 vérifications stables consécutives. S'il reste des cases
        # changées non expliquées (ex. le coup du joueur pas encore bien visible),
        # on attend plus longtemps (~0,7 s) pour laisser la suite à 2 coups apparaître
        # au lieu de figer une référence qui contiendrait déjà un coup non compté.
        unexplained = any(sq not in best_squares for sq in significant)
        if self.consecutive_match_count < (6 if unexplained else 2):
            return
        self.consecutive_match_count = 0
        self.ambiguity = None
        self.last_candidate_move = None

        bot_first = self.state == "WAIT_BOT_EXECUTION"
        for i, m in enumerate(best_line):
            if self.state == "GAME_OVER":
                break
            player = "bot" if (bot_first and i == 0) else "human"
            if player == "bot":
                self.log(f"🤖 Coup du bot {m.uci()} exécuté physiquement avec succès !")
            self._apply_move(m, cur_warped, player=player)
        if len(best_line) > 1:
            self.log("⚡ Coup du bot et coup du joueur détectés ensemble.")

    def _apply_move(self, move, cur_warped, player="human"):
        san = self.board.san(move)
        uci = move.uci()
        self.board.push(move)
        if cur_warped is not None:
            self.reference_warped = cur_warped.copy()
        self.last_move = uci
        # Un coup vient d'être joué : la flèche (coup du bot ou conseil) n'a plus lieu d'être
        self.ambiguity = None
        self.pending_bot_move = None
        self.vision.ar_arrow = None

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
            self._save_game()
            self.log(f"🏁 Fin de partie : {self.board.result()}")
            return

        # Tour suivant
        if self.engine.game_mode == "HUMAN_VS_BOT" and self.board.turn != self.engine.human_color:
            self._start_bot_turn()
        else:
            self.state = "WAIT_HUMAN_MOVE"

        # Sauvegarde automatique de l'état de la partie
        self._save_game()

    def _start_bot_turn(self):
        """Passe au tour du bot et lance son calcul en arrière-plan (appelé sous verrou)."""
        self.state = "WAIT_BOT_THINKING"
        threading.Thread(target=self._bot_turn_thread,
                         args=(self._generation, self.board.copy()), daemon=True).start()

    def _bot_turn_thread(self, generation, board):
        """Thread asynchrone pour calculer le coup du Bot sans geler la vidéo."""
        time.sleep(0.3)
        res = self.engine.compute_move(board)
        if not res:
            return

        with self._lock:
            # Partie relancée ou coup annulé pendant le calcul : résultat obsolète
            if (generation != self._generation or self.state != "WAIT_BOT_THINKING"
                    or board.fen() != self.board.fen()):
                return
            move = res["move"]
            is_capture = self.board.is_capture(move)
            self.pending_bot_move = {
                "move": move,
                "uci": res["uci"],
                "san": res["san"],
                "is_capture": is_capture,
                "is_castling": self.board.is_castling(move)
            }
            # Afficher la flèche AR
            self.vision.ar_arrow = {
                "from_sq": move.from_square,
                "to_sq": move.to_square,
                "is_capture": is_capture,
                "color": (0, 200, 255) if not is_capture else (40, 50, 240),
                "label": f"Bot: {res['san']}",
                "kind": "bot"
            }
            self.consecutive_match_count = 0
            self.last_candidate_move = None
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
                self._generation += 1
                self.ambiguity = None
                self.consecutive_match_count = 0
                self.last_candidate_move = None
                self.state = "WAIT_HUMAN_MOVE"
                self._save_game()
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
                "arrow": self._arrow_status(),
                "ambiguity": self.ambiguity,
                "history": self.move_history[-10:]
            }

    def _arrow_status(self):
        """Flèche actuellement affichée (coup du bot ou conseil), partagée par les trois vues."""
        a = self.vision.ar_arrow
        if not a:
            return None
        return {
            "uci": chess.square_name(a["from_sq"]) + chess.square_name(a["to_sq"]),
            "kind": a.get("kind", "bot")
        }
