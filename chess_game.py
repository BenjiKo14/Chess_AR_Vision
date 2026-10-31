import sys
import time
import json
import os
import threading
import chess
import cv2
import numpy as np

from board_reader import BoardReader

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
        # Garde-fous contre les coups fantômes (ombre, main immobile, reflet)
        self.confirm_seconds = 0.7    # le même coup doit être vu pendant ce temps sur des images stables
        self.min_score_margin = 2.0   # avance minimale sur un autre coup possible
        self.leak_ratio = 0.7         # débordement toléré d'une pièce haute sur une case voisine
        self.same_piece_ncc = 0.7     # au-dessus : la case « vidée » montre encore la même pièce
        self.empty_max_std = 6.0      # texture max d'une case vide (sous une ombre, une pièce reste au-dessus)
        self.hint_after_seconds = 1.5
        self._candidate_since = None
        self._blocked_reason = None
        self._blocked_since = None
        self.vision_hint = None       # explication affichée quand un coup reste bloqué
        # Lecture directe de la position sur l'image (indépendante de la référence)
        self.reader = BoardReader()
        self.reference_version = None
        self.max_changed_for_reference = 32   # au-delà (moitié du plateau) : lumière ou caméra changée
        self.diff_bonus = 1.0                 # poids d'une case qui a bien changé par rapport à la référence
        self.frame_veto = 1.5                 # avec référence : l'image ne doit pas contredire nettement une case
        self.frame_support = 1.4              # sans référence : chaque case lisible doit confirmer (≈ 4 contre 1)
        self.accept_with_reference = 1.5
        self.accept_frame_only = 4.0
        self.max_disagree_for_rebase = 3      # sans écarts mémorisés (après redémarrage) : tolérance pour les débordements
        self.max_disagree_for_resync = 4      # au-delà, « Resync » est refusé : l'échiquier ne correspond pas
        self.frame_confirm = 0.8              # l'image confirme nettement le contenu attendu d'une case (hors cases du coup)
        self._baseline_disagree = None
        self._noise_since = None
        self.rebase_seconds = 1.0
        self.orientation_check_seconds = 3.0
        self.orientation_gain = 15.0          # gain de vraisemblance exigé pour tourner la calibration
        self._orientation_checked_at = 0.0
        self._orientation_votes = (0, 0)
        self._none_since = None
        self._cand_cache_key = None
        self._cand_cache = []
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
                # Partie en cours : reprendre au bon tour (le bot recalcule son coup si c'est à lui)
                if saved_state in ("WAIT_HUMAN_MOVE", "WAIT_BOT_THINKING", "WAIT_BOT_EXECUTION"):
                    if self.engine.game_mode == "HUMAN_VS_BOT" and self.board.turn != self.engine.human_color:
                        self._start_bot_turn()
                    else:
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
            self._reset_candidate()
            self._clear_block()
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
            # Ne pas écraser ni réinitialiser si une partie est déjà en cours
            if self.state not in ("SETUP_CHECK", "IDLE"):
                return False, f"Impossible de valider les pièces initiales : la partie est déjà en cours (état : {self.state})"

            cur = self.vision.get_latest_warped_frame()
            if cur is None:
                return False, "Aucune image reçue de la caméra"

            self._set_reference(cur)
            self.vision.setup_mode = False

            # Déterminer qui commence
            if self.engine.game_mode == "HUMAN_VS_BOT" and self.board.turn != self.engine.human_color:
                self._start_bot_turn()
            else:
                self.state = "WAIT_HUMAN_MOVE"

            self.log("✅ Plateau validé. La partie commence !")
            self._save_game()
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
            self._reset_candidate()
            self._clear_block()
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
                self._set_reference(cur)
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
        """Recalibre l'image de référence sur l'état visuel actuel (J'adoube / lumière).
        Refusé si l'échiquier ne correspond manifestement pas à la position affichée :
        sinon un désordre deviendrait la référence et plus aucun coup ne serait compris."""
        with self._lock:
            cur = self.vision.get_latest_warped_frame()
            if cur is None:
                return False, "Aucune image reçue de la caméra"
            probs = self.reader.read(cur, self.board)
            disagree, _ = self._frame_disagreement(probs)
            if len(disagree) > self.max_disagree_for_resync:
                msg = (f"Resync refusé : l'échiquier ne correspond pas à l'état affiché "
                       f"({self._names(disagree)}). Remettez les pièces ou utilisez « ✏️ Corriger la position ».")
                self.log(f"⚠️ {msg}")
                return False, msg
            self._set_reference(cur, probs)
            self._reset_candidate()
            self._clear_block()
            self.log("📸 Image de référence resynchronisée.")
            return True, "Image de référence synchronisée"

    def _set_reference(self, frame, probs=None):
        """Image de référence = dernière image où la position connue est vraie.
        On mémorise aussi les cases où la lecture de l'image diffère « normalement » de la
        position (pièce haute qui déborde, pièce noire peu visible) : seuls ces écarts-là
        seront tolérés pour un recalage automatique."""
        self.reference_warped = frame.copy() if frame is not None else None
        self.reference_version = self.vision.geometry_version
        self._none_since = None
        self._noise_since = None
        if frame is None:
            self._baseline_disagree = None
            return
        if probs is None:
            probs = self.reader.read(frame, self.board)
        self._baseline_disagree = self._frame_disagreement(probs)[0]

    def _frame_disagreement(self, probs):
        """Cases où l'image contredit nettement la position connue, et parmi elles celles
        censées occupées qui paraissent vides."""
        idx = self._class_index_board(self.board)
        p_label = probs[np.arange(64), idx]
        disagree = {sq for sq in chess.SQUARES if p_label[sq] < 0.1}
        vacated = {sq for sq in disagree if idx[sq] != 0 and probs[sq, 0] > 0.5}
        return disagree, vacated

    def _reference_usable(self):
        return (self.reference_warped is not None
                and self.reference_version == self.vision.geometry_version)

    def update_vision_tick(self):
        """Appelé périodiquement : lit la position sur l'image et la compare à la position connue."""
        with self._lock:
            if self.state not in ("WAIT_HUMAN_MOVE", "WAIT_BOT_EXECUTION"):
                return
            if not self.vision.is_board_stable:
                # Une main repasse au-dessus du plateau : la confirmation repart de zéro
                self._reset_candidate()
                self._none_since = None
                return
            cur = self.vision.get_latest_warped_frame()
            if cur is None:
                return
            board = self.board.copy(stack=False)
            fen = board.fen()

        # Lecture de l'image hors verrou (quelques dizaines de ms)
        probs = self.reader.read(cur, board)
        self.vision.last_detected_pieces = probs
        if self._maybe_fix_orientation(cur, board, probs):
            return

        with self._lock:
            if self.board.fen() != fen or self.state not in ("WAIT_HUMAN_MOVE", "WAIT_BOT_EXECUTION"):
                return  # la partie a bougé pendant la lecture
            changes = None
            if self._reference_usable():
                changes = self.vision.compute_all_square_changes(self.reference_warped, cur)
                n_changed = sum(1 for v in changes.values() if v >= self.min_change_threshold)
                if n_changed > self.max_changed_for_reference:
                    # Lumière ou caméra différente : la référence ne sert plus, on lit l'image seule
                    changes = None
            self._update_debug_info(changes)
            self._decide(probs, changes, cur)

    def _update_debug_info(self, changes):
        if not changes:
            self.vision.top_diff_info = "" if changes is not None else "Lecture directe (référence périmée)"
            return
        sorted_sq = sorted(changes.items(), key=lambda x: x[1], reverse=True)
        if sorted_sq[0][1] > 6.0:
            (s1, v1), (s2, v2) = sorted_sq[0], sorted_sq[1]
            self.vision.top_diff_info = f"Top diff: {chess.square_name(s1)}({v1}), {chess.square_name(s2)}({v2})"
        else:
            self.vision.top_diff_info = ""

    def _maybe_fix_orientation(self, cur, board, probs):
        """Si le plateau est lu tourné (90°/180°), les pièces blanches n'apparaissent pas du
        bon côté : la position connue s'explique alors bien mieux en tournant l'image.
        On tourne la calibration après deux vérifications concordantes."""
        now = time.time()
        if now - self._orientation_checked_at < self.orientation_check_seconds:
            return False
        self._orientation_checked_at = now
        idx = self._class_index_board(board)
        ll_now = float(np.sum(np.log(probs[np.arange(64), idx])))
        best_k, best_ll = 0, ll_now
        img = cur
        for k in (1, 2, 3):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            p = self.reader.read(img, board)
            ll = float(np.sum(np.log(p[np.arange(64), idx])))
            if ll > best_ll:
                best_k, best_ll = k, ll
        if best_k == 0 or best_ll - ll_now < self.orientation_gain:
            self._orientation_votes = (0, 0)
            return False
        k_prev, votes = self._orientation_votes
        votes = votes + 1 if k_prev == best_k else 1
        self._orientation_votes = (best_k, votes)
        if votes < 2:
            return False
        with self._lock:
            for _ in range(best_k):
                self.vision.rotate_board_90()
            self._orientation_votes = (0, 0)
            self._reset_candidate()
        self.log(f"🔄 Plateau lu à l'envers : orientation corrigée automatiquement ({best_k * 90}°).")
        return True

    @staticmethod
    def _piece_class(piece):
        if piece is None:
            return 0
        return 1 if piece.color == chess.WHITE else 2

    def _class_index_board(self, board):
        return np.array([self._piece_class(board.piece_at(s)) for s in chess.SQUARES])

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

    def _reset_candidate(self):
        self.consecutive_match_count = 0
        self.last_candidate_move = None
        self._candidate_since = None

    def _block(self, reason):
        """Aucun coup n'est validé tant que la scène n'est pas propre. Si l'attente dure,
        on l'explique dans le bandeau pour que le joueur sache quoi faire."""
        self._reset_candidate()
        now = time.time()
        if self._blocked_reason != reason:
            self._blocked_reason = reason
            self._blocked_since = now
            self.vision_hint = None
        elif self.vision_hint is None and now - self._blocked_since >= self.hint_after_seconds:
            self.vision_hint = reason
            self.log(f"⏸️ Coup non validé : {reason}")

    def _clear_block(self):
        self._blocked_reason = None
        self._blocked_since = None
        self.vision_hint = None

    @staticmethod
    def _names(squares):
        return ", ".join(chess.square_name(sq) for sq in sorted(squares))

    def _check_line_is_clean(self, line, expected, changes, significant, cur_warped):
        """Double vérification avant de valider une suite de coups.
        Retourne None si tout est cohérent, sinon la raison du refus."""
        # Main / bras au-dessus du plateau ou changement de lumière : trop de cases bougent
        if len(significant) > len(expected) + 3:
            return f"trop de cases changent ({len(significant)}) : main au-dessus du plateau ou lumière ?"

        # Cases changées non expliquées par le coup. Seul un « débordement » est toléré :
        # une pièce haute qui empiète sur la case voisine (adjacente et nettement plus faible).
        weakest = min(changes.get(sq, 0.0) for sq in expected)
        stray = []
        for sq in significant:
            if sq in expected:
                continue
            adjacent = any(chess.square_distance(sq, e) == 1 for e in expected)
            if not (adjacent and changes.get(sq, 0.0) < self.leak_ratio * weakest):
                stray.append(sq)
        if stray:
            return (f"changement inexpliqué sur {self._names(stray)}. Si l'échiquier correspond à l'état "
                    f"affiché → « Resync », sinon « ✏️ Corriger la position »")

        # Les cases que le coup doit vider ne doivent pas montrer la même pièce qu'avant
        # (une ombre ou un reflet assombrit la pièce mais garde sa forme).
        after = self.board.copy(stack=False)
        for m in line:
            after.push(m)
        vacated = [sq for sq in expected if self.board.piece_at(sq) and not after.piece_at(sq)]
        structure = self.vision.compute_square_structure(self.reference_warped, cur_warped, vacated)
        still_there = [sq for sq, (ncc, _std_ref, std_cur) in structure.items()
                       if ncc >= self.same_piece_ncc and std_cur >= self.empty_max_std]
        if still_there:
            return f"la pièce semble encore sur {self._names(still_there)} (ombre ou reflet ?)"
        return None

    def _candidate_entries(self):
        """Hypothèses (aucun coup, 1 ou 2 demi-coups) avec, pour chacune, les cases qui
        changent : (case, classe avant, classe après) avec 0 = vide, 1 = blanc, 2 = noir.
        Mis en cache tant que la position et l'état ne changent pas."""
        key = (self.board.fen(), self.state,
               self.pending_bot_move["uci"] if self.pending_bot_move else None)
        if self._cand_cache_key == key:
            return self._cand_cache
        before = self.board.piece_map()
        entries = []
        for line in self._candidate_lines():
            after = self.board.copy(stack=False)
            for m in line:
                after.push(m)
            amap = after.piece_map()
            diffs = [(sq, self._piece_class(before.get(sq)), self._piece_class(amap.get(sq)))
                     for sq in set(before) | set(amap) if before.get(sq) != amap.get(sq)]
            entries.append((line, diffs))
        self._cand_cache_key, self._cand_cache = key, entries
        return entries

    def _evaluate(self, diffs, logp, changes, significant=()):
        """Preuves en faveur d'une hypothèse, relativement à « aucun coup ».
        Retourne (score, pire_case, pire_contradiction, nb_cases_lisibles, ok_diff).
        pire_contradiction ne retient que ce que l'image voit de façon fiable : une pièce
        encore présente sur une case censée être vidée, ou une pièce de la mauvaise couleur.
        (« Case qui paraît vide » n'est pas fiable : pièce noire sur case foncée sous lampe faible.)"""
        score, worst, contra, informative = 0.0, np.inf, np.inf, 0
        ok_diff = True
        for sq, old, new in diffs:
            r = 0.0
            if old != new:
                r = float(logp[sq, new] - logp[sq, old])
                informative += 1
                worst = min(worst, r)
                if new == 0:
                    contra = min(contra, r)
                else:
                    opposite = 3 - new
                    contra = min(contra, float(logp[sq, new] - logp[sq, opposite]))
            if changes is not None:
                if changes.get(sq, 0.0) >= self.min_change_threshold:
                    r += self.diff_bonus
                elif old != new:
                    ok_diff = False   # l'image de référence dit que cette case n'a pas bougé
                else:
                    # Pièce remplacée par une autre de même couleur (ex. reprise) :
                    # le changement peut être discret, simple pénalité
                    r -= self.diff_bonus
            score += r
        if changes is not None and diffs:
            # Cases qui ont changé sans que l'hypothèse les explique (hors débordement
            # d'une pièce haute sur une case voisine, nettement plus faible). On ne compte
            # pas celles que l'image montre conformes (différence venant de la référence).
            touched = {sq for sq, _, _ in diffs}
            weakest = min(changes.get(sq, 0.0) for sq in touched)
            for sq in significant:
                if sq in touched:
                    continue
                v = changes.get(sq, 0.0)
                leak = (v < self.leak_ratio * weakest
                        and any(chess.square_distance(sq, t) == 1 for t in touched))
                if not leak:
                    score -= self.diff_bonus
        return score, worst, contra, informative, ok_diff

    def _decide(self, probs, changes, cur):
        """Choisit l'hypothèse la mieux soutenue par l'image. N'invente rien : un coup n'est
        validé que si chaque case qu'il modifie le confirme, s'il devance nettement les
        autres hypothèses et s'il reste le même pendant confirm_seconds."""
        logp = np.log(probs)
        ref_ok = changes is not None
        raw_significant = [sq for sq, v in (changes or {}).items() if v >= self.min_change_threshold]
        significant = raw_significant
        if ref_ok:
            # Hors des cases du coup étudié, une case que l'image montre nettement conforme
            # à la position connue n'a pas « changé » : la différence vient de la référence
            # (ombre, pièce bousculée puis remise, référence prise au mauvais moment).
            # Les cases du coup lui-même gardent leur différence brute : une pièce noire qui
            # arrive sur une case foncée peut être lue « vide » sous une lampe faible.
            idx = self._class_index_board(self.board)
            confirmed = probs[np.arange(64), idx] >= self.frame_confirm
            significant = [sq for sq in raw_significant if not confirmed[sq]]

        # Avec une référence valable et rien qui bouge : rien à faire
        if ref_ok and len(raw_significant) < 2:
            self._reset_candidate()
            self._clear_block()
            self._none_since = None
            self._noise_since = None
            return

        scored = []
        best_any = []   # toutes les hypothèses avec des preuves positives, valides ou non
        for line, diffs in self._candidate_entries():
            score, worst, contra, informative, ok_diff = self._evaluate(diffs, logp, changes, significant)
            if ref_ok:
                # Référence fiable : chaque case touchée doit avoir changé, et l'image
                # ne doit contredire nettement aucune d'elles
                valid = ok_diff and informative > 0 and contra >= -self.frame_veto
                threshold = self.accept_with_reference
            else:
                # Lecture directe seule : chaque case lisible doit confirmer clairement
                valid = informative > 0 and worst >= self.frame_support
                threshold = self.accept_frame_only
            if valid and score >= threshold:
                scored.append((score, line, diffs))
            elif score > 0 and informative > 0 and (ok_diff or not ref_ok):
                # Écartée faute de preuve assez nette dans l'image (pas démentie par la
                # référence) : elle reste une rivale crédible pour la marge
                best_any.append(score)

        if not scored:
            if ref_ok and len(significant) < 2:
                # Les seules différences viennent de la référence : on la nettoie
                self._reset_candidate()
                self._try_rebase(probs, cur, "📸 Référence nettoyée (elle ne correspondait plus à la position).",
                                 timer="_noise_since")
                return
            self._handle_no_move(probs, changes, significant, cur)
            return
        self._none_since = None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line, best_diffs = scored[0]

        # Hypothèses indiscernables : mêmes cases, mêmes contenus visibles (ex. reprendre
        # sur la case d'arrivée du bot ou aller sur sa case de départ). On ne devine pas.
        def signature(diffs):
            return frozenset((sq, new) for sq, old, new in diffs), frozenset(sq for sq, _, _ in diffs)
        sig = signature(best_diffs)
        rivals = [l for _, l, d in scored[1:] if signature(d) == sig]
        if rivals:
            options = sorted(" puis ".join(m.uci() for m in l) for l in [best_line] + rivals)
            key = ("ambiguous",) + tuple(options)
            if self.last_candidate_move != key:  # ne logguer qu'une fois
                self._reset_candidate()
                self.last_candidate_move = key
                self.ambiguity = " ou ".join(options)
                self.log(f"❓ Position ambiguë ({' ou '.join(options)}) : placez la bonne position via « ✏️ Corriger la position ».")
            return
        self.ambiguity = None

        best_squares = {sq for sq, _, _ in best_diffs}
        if ref_ok:
            reason = self._check_line_is_clean(best_line, best_squares, changes, significant, cur)
            if reason:
                self._block(reason)
                return

        runner_up = max([s for s, _, _ in scored[1:]] + best_any, default=None)
        if runner_up is not None and best_score - runner_up < self.min_score_margin:
            self._block("deux coups possibles trop proches, attente d'une image plus nette")
            return
        self._clear_block()

        key = tuple(m.uci() for m in best_line)
        now = time.time()
        if self.last_candidate_move == key:
            self.consecutive_match_count += 1
        else:
            self.last_candidate_move = key
            self.consecutive_match_count = 1
            self._candidate_since = now

        # Confirmation : même coup vu sur plusieurs images stables pendant confirm_seconds
        if self.consecutive_match_count < 3 or now - self._candidate_since < self.confirm_seconds:
            return
        self._reset_candidate()
        if not ref_ok:
            self.log("👁️ Coup lu directement sur l'image (référence périmée : lumière ou caméra changée).")

        bot_first = self.state == "WAIT_BOT_EXECUTION"
        for i, m in enumerate(best_line):
            if self.state == "GAME_OVER":
                break
            player = "bot" if (bot_first and i == 0) else "human"
            if player == "bot":
                self.log(f"🤖 Coup du bot {m.uci()} exécuté physiquement avec succès !")
            self._apply_move(m, cur, player=player)
        if len(best_line) > 1:
            self.log("⚡ Coup du bot et coup du joueur détectés ensemble.")

    def _handle_no_move(self, probs, changes, significant, cur):
        """Aucun coup n'est soutenu par l'image."""
        self._reset_candidate()
        if changes is not None and len(significant) > 6:
            self._block(f"trop de cases changent ({len(significant)}) : main au-dessus du plateau ou ombre ?")
            return
        if changes is not None:
            # Référence fiable mais des cases ont changé sans correspondre à un coup
            self._block(f"changement sur {self._names(significant)} qui ne correspond à aucun coup légal. "
                        f"Si l'échiquier correspond à l'état affiché → « Resync », sinon « ✏️ Corriger la position »")
            return

        # Référence absente ou périmée (caméra reconnectée, lumière, recalibrage) : si l'image
        # correspond à la position connue, on reprend une référence propre automatiquement.
        self._try_rebase(probs, cur, "📸 Référence recalée automatiquement (lumière ou caméra changée).")

    def _try_rebase(self, probs, cur, message, timer="_none_since"):
        """Reprend l'image actuelle comme référence, seulement si elle montre la position
        connue (aux écarts habituels près) pendant rebase_seconds."""
        disagree, vacated = self._frame_disagreement(probs)
        baseline = self._baseline_disagree
        if baseline is None:
            allowed = not vacated and len(disagree) <= self.max_disagree_for_rebase
        else:
            # Seuls les écarts déjà présents quand la position était sûre sont tolérés :
            # une pièce déplacée (même une seule) empêche le recalage.
            allowed = disagree <= baseline
        if not allowed:
            setattr(self, timer, None)
            known = baseline or set()
            new = sorted(disagree - known)
            vac = sorted(vacated - known)
            if len(disagree) > 10:
                self._block(f"l'image ne correspond pas à la position ({len(disagree)} cases diffèrent) : "
                            f"pièces déplacées ou calibration à refaire")
            elif vac:
                self._block(f"{self._names(vac)} paraît vide alors qu'une pièce devrait y être. "
                            f"Remettez la position affichée, ou « ✏️ Corriger la position »")
            else:
                self._block(f"position vue différente sur {self._names(new or disagree)}, sans coup légal qui "
                            f"l'explique. Remettez la position affichée, ou « ✏️ Corriger la position »")
            return
        now = time.time()
        since = getattr(self, timer)
        if since is None:
            setattr(self, timer, now)
        elif now - since >= self.rebase_seconds:
            had_reference = self.reference_warped is not None
            self._set_reference(cur, probs)
            self._clear_block()
            if had_reference:
                self.log(message)

    def _apply_move(self, move, cur_warped, player="human"):
        san = self.board.san(move)
        uci = move.uci()
        self.board.push(move)
        if cur_warped is not None:
            self._set_reference(cur_warped)
        self.last_move = uci
        # Un coup vient d'être joué : la flèche (coup du bot ou conseil) n'a plus lieu d'être
        self.ambiguity = None
        self._reset_candidate()
        self._clear_block()
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
                    self._set_reference(cur)
                self.last_move = self.move_history[-1]["uci"] if self.move_history else None
                self.vision.ar_arrow = None
                self.vision.highlight_squares.clear()
                self.pending_bot_move = None
                self._generation += 1
                self.ambiguity = None
                self._reset_candidate()
                self._clear_block()
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
                "vision_hint": self.vision_hint,
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
