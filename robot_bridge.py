import sys
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

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king"
}

PIECE_PROFILES = {
    "pawn":   {"grasp_z_mm": 22.0, "lift_z_mm": 85.0, "gripper_open": 55, "gripper_close": 22},
    "knight": {"grasp_z_mm": 30.0, "lift_z_mm": 90.0, "gripper_open": 60, "gripper_close": 25},
    "bishop": {"grasp_z_mm": 34.0, "lift_z_mm": 95.0, "gripper_open": 60, "gripper_close": 24},
    "rook":   {"grasp_z_mm": 28.0, "lift_z_mm": 90.0, "gripper_open": 62, "gripper_close": 26},
    "queen":  {"grasp_z_mm": 42.0, "lift_z_mm": 105.0, "gripper_open": 65, "gripper_close": 25},
    "king":   {"grasp_z_mm": 48.0, "lift_z_mm": 110.0, "gripper_open": 68, "gripper_close": 26},
}

class RobotChessBridge:
    """
    Traducteur de coups d'échecs en séquences d'actions physiques atomiques
    pour le bras robotique SO-101.
    """
    def __init__(self, square_size_mm=45.0, log_fn=None):
        self.square_size_mm = square_size_mm
        self.log_fn = log_fn or safe_log_print
        self.graveyard_counter = 0

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

    def get_square_spatial_info(self, sq, vision_manager):
        """
        Retourne les coordonnées d'une case dans le repère caméra (pixels uv),
        le repère 512x512 redressé (warped xy) et le repère physique en millimètres.
        """
        col = sq % 8
        row = 7 - (sq // 8)

        # Repère 512x512
        wx = (col + 0.5) * 64.0
        wy = (row + 0.5) * 64.0

        # Repère millimétrique (centre de l'échiquier = 0, 0)
        x_mm = (col - 3.5) * self.square_size_mm
        y_mm = (3.5 - row) * self.square_size_mm

        # Projection sur l'image caméra
        cam_uv = vision_manager.warped_points_to_camera([(wx, wy)])[0]

        return {
            "square": chess.square_name(sq),
            "col": col,
            "row": row,
            "warped_xy": [round(wx, 1), round(wy, 1)],
            "camera_uv": [round(cam_uv[0], 1), round(cam_uv[1], 1)],
            "board_mm": [round(x_mm, 1), round(y_mm, 1)]
        }

    def build_robot_action_plan(self, board_before_move, move, vision_manager):
        """
        Décompose un coup en liste ordonnée d'actions atomiques :
        [REMOVE_CAPTURED (si prise) -> MOVE_PIECE -> RETURN_HOME]
        """
        actions = []
        is_capture = board_before_move.is_capture(move)
        is_castling = board_before_move.is_castling(move)
        is_en_passant = board_before_move.is_en_passant(move)

        moving_piece = board_before_move.piece_at(move.from_square)
        moving_name = PIECE_NAMES.get(moving_piece.piece_type, "pawn") if moving_piece else "pawn"

        # 1. Gestion de la pièce capturée
        if is_capture:
            if is_en_passant:
                cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                cap_piece = board_before_move.piece_at(cap_sq)
            else:
                cap_sq = move.to_square
                cap_piece = board_before_move.piece_at(cap_sq)

            cap_name = PIECE_NAMES.get(cap_piece.piece_type, "pawn") if cap_piece else "pawn"
            cap_coords = self.get_square_spatial_info(cap_sq, vision_manager)

            graveyard_x_mm = 5.0 * self.square_size_mm  # Bord droit
            graveyard_y_mm = (self.graveyard_counter % 8 - 3.5) * self.square_size_mm
            self.graveyard_counter += 1

            actions.append({
                "action": "REMOVE_CAPTURED",
                "piece": cap_name,
                "profile": PIECE_PROFILES.get(cap_name),
                "from_square": chess.square_name(cap_sq),
                "from_coords": cap_coords,
                "to_graveyard_mm": [round(graveyard_x_mm, 1), round(graveyard_y_mm, 1)]
            })

        # 2. Déplacement de la pièce principale
        from_coords = self.get_square_spatial_info(move.from_square, vision_manager)
        to_coords = self.get_square_spatial_info(move.to_square, vision_manager)

        actions.append({
            "action": "MOVE_PIECE",
            "piece": moving_name,
            "profile": PIECE_PROFILES.get(moving_name),
            "from_square": chess.square_name(move.from_square),
            "from_coords": from_coords,
            "to_square": chess.square_name(move.to_square),
            "to_coords": to_coords
        })

        # 3. Deuxième mouvement en cas de roque (déplacement de la Tour)
        if is_castling:
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) == 6:  # Petit roque
                r_from = chess.square(7, rank)
                r_to = chess.square(5, rank)
            else:  # Grand roque
                r_from = chess.square(0, rank)
                r_to = chess.square(3, rank)

            actions.append({
                "action": "MOVE_ROOK_CASTLING",
                "piece": "rook",
                "profile": PIECE_PROFILES.get("rook"),
                "from_square": chess.square_name(r_from),
                "from_coords": self.get_square_spatial_info(r_from, vision_manager),
                "to_square": chess.square_name(r_to),
                "to_coords": self.get_square_spatial_info(r_to, vision_manager)
            })

        # 4. Retour obligatoire en position HOME
        actions.append({
            "action": "RETURN_HOME",
            "note": "Libère le champ de vision au-dessus de l'échiquier"
        })

        return {
            "move_uci": move.uci(),
            "san": board_before_move.san(move),
            "step_count": len(actions),
            "actions": actions
        }
