import os
import sys
import time
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
try:
    from flask_cors import CORS
    HAVE_CORS = True
except ImportError:
    HAVE_CORS = False

from camera import CameraManager
from chess_vision import ChessVisionManager
from chess_engine import ChessEngineManager
from chess_game import ChessGameManager
from robot_bridge import RobotChessBridge
from dataset_collector import DatasetCollector
from onnx_classifier import OnnxPieceClassifier

app = Flask(__name__)
if HAVE_CORS:
    CORS(app)

logs_history = []
def log(msg):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    try:
        print(entry)
    except UnicodeEncodeError:
        try:
            enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
            print(str(entry).encode(enc, errors='replace').decode(enc))
        except Exception:
            try:
                print(str(entry).encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
    except Exception:
        pass

    logs_history.append(entry)
    if len(logs_history) > 100:
        logs_history.pop(0)

# Instances des briques du système
camera_mgr = CameraManager(log_fn=log)
vision_mgr = ChessVisionManager(log_fn=log)
camera_mgr.frame_processor = vision_mgr.process_frame

# Reconnexion automatique de la caméra sauvegardée (index, rotation 180, miroir)
try:
    if getattr(camera_mgr, 'was_running', True):
        saved_idx = getattr(camera_mgr, 'camera_index', 0)
        if saved_idx is not None and saved_idx != "photo":
            log(f"📷 Reconnexion automatique caméra {saved_idx} (180°={camera_mgr.rotate_180}, Miroir={camera_mgr.flip_h})...")
            ok_cam, err_cam = camera_mgr.start(index=saved_idx)
            if not ok_cam and saved_idx != 0:
                log(f"[WARN] Caméra {saved_idx} non joignable ({err_cam}), repli caméra 0...")
                camera_mgr.start(index=0)
except Exception as e:
    log(f"[WARN] Erreur démarrage auto caméra: {e}")

engine_mgr = ChessEngineManager(log_fn=log)
game_mgr = ChessGameManager(vision_mgr, engine_mgr, log_fn=log)
robot_bridge = RobotChessBridge(log_fn=log)
dataset_collector = DatasetCollector(log_fn=log)
onnx_classifier = OnnxPieceClassifier(log_fn=log)

# Thread de détection périodique des coups d'échecs
def _vision_tick_loop():
    while True:
        try:
            if camera_mgr.is_running:
                game_mgr.update_vision_tick()
        except Exception as e:
            pass
        time.sleep(0.12)

threading.Thread(target=_vision_tick_loop, daemon=True).start()

# --- Routes Page Web ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/logs')
def get_logs():
    return jsonify({'success': True, 'logs': logs_history[-30:]})

# --- Routes Caméra (Slice 1) ---
@app.route('/api/camera/list')
def camera_list():
    cameras = camera_mgr.list_cameras()
    return jsonify({'success': True, 'cameras': cameras, 'status': camera_mgr.get_status()})

@app.route('/api/camera/status')
def camera_status():
    return jsonify({'success': True, **camera_mgr.get_status()})

@app.route('/api/camera/connect', methods=['POST'])
def camera_connect():
    data = request.get_json(silent=True) or {}
    idx = int(data.get('index', 0))
    ok, err = camera_mgr.start(index=idx)
    vision_mgr.invalidate_geometry()  # nouvelle image : l'ancienne référence n'est plus comparable
    if ok:
        return jsonify({'success': True, 'status': camera_mgr.get_status()})
    return jsonify({'success': False, 'error': err}), 400

@app.route('/api/camera/disconnect', methods=['POST'])
def camera_disconnect():
    camera_mgr.stop()
    return jsonify({'success': True, 'status': camera_mgr.get_status()})

@app.route('/api/camera/settings', methods=['POST'])
def camera_settings():
    data = request.get_json(silent=True) or {}
    vision_mgr.invalidate_geometry()
    status = camera_mgr.update_settings(
        rotate_180=data.get('rotate_180'),
        flip_h=data.get('flip_h'),
        toggle_flip_h=bool(data.get('toggle_flip_h')),
        toggle_rotate_180=bool(data.get('toggle_rotate_180')),
        show_crosshair=data.get('show_crosshair'),
        active_filters=data.get('active_filters')
    )
    return jsonify({'success': True, 'status': status})

@app.route('/api/camera/stream')
def camera_stream():
    def generate():
        while camera_mgr.is_running:
            jpeg = camera_mgr.get_jpeg_frame()
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/snapshot', methods=['POST'])
def camera_snapshot():
    ok, filename, filepath = camera_mgr.take_snapshot()
    if ok:
        return jsonify({'success': True, 'filename': filename, 'url': f'/snapshots/{filename}'})
    return jsonify({'success': False, 'error': filename}), 400

@app.route('/api/camera/upload-photo', methods=['POST'])
def camera_upload_photo():
    if 'photo' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier photo fourni'}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nom de fichier vide'}), 400

    file_bytes = file.read()
    ok, err = camera_mgr.load_static_image(file_bytes)
    vision_mgr.invalidate_geometry()  # nouvelle image : l'ancienne référence n'est plus comparable
    if ok:
        return jsonify({'success': True, 'status': camera_mgr.get_status()})
    return jsonify({'success': False, 'error': err}), 400

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory('snapshots', filename)

def continuous_detection_worker():
    while True:
        time.sleep(1.0)
        if not getattr(vision_mgr, 'continuous_detection', False):
            continue
        now = time.time()
        last_time = getattr(vision_mgr, 'last_continuous_detection_time', 0.0)
        if now - last_time < 5.0:
            continue
        if not camera_mgr.is_running:
            continue
        raw_frame = camera_mgr.get_raw_frame()
        if raw_frame is None:
            continue
        # Ne pas réajuster si la main est au-dessus du plateau ou si un coup est en cours de validation
        if not vision_mgr.is_board_stable or getattr(game_mgr, 'consecutive_match_count', 0) > 0:
            continue
        try:
            old_version = vision_mgr.geometry_version
            ok, msg, corners = vision_mgr.auto_detect_board(raw_frame)
            vision_mgr.last_continuous_detection_time = time.time()
            # Logger seulement si un recalage réel a eu lieu
            if ok and vision_mgr.geometry_version != old_version:
                log(f"[Auto-Detection Continue 5s - Recalage] {msg}")
        except Exception as e:
            log(f"[Auto-Detection Continue Erreur] {e}")

continuous_thread = threading.Thread(target=continuous_detection_worker, daemon=True)
continuous_thread.start()

# --- Routes Vision & Calibration (Slice 1 & 2) ---
@app.route('/api/vision/status')
def vision_status():
    return jsonify({'success': True, **vision_mgr.get_status()})

@app.route('/api/vision/continuous-detection', methods=['POST'])
def vision_continuous_detection():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    vision_mgr.continuous_detection = enabled
    if enabled:
        vision_mgr.last_continuous_detection_time = 0.0
    vision_mgr.save_calibration()
    log(f"[Vision] Détection continue (5s) : {'ACTIVÉE' if enabled else 'DÉSACTIVÉE'}")
    return jsonify({'success': True, 'enabled': vision_mgr.continuous_detection})

@app.route('/api/vision/auto-calibrate', methods=['POST'])
def vision_auto_calibrate():
    raw_frame = camera_mgr.get_raw_frame()
    if raw_frame is None:
        return jsonify({'success': False, 'error': 'Aucune image reçue de la caméra'}), 400
    ok, msg, corners = vision_mgr.auto_detect_board(raw_frame)
    return jsonify({
        'success': ok,
        'message': msg,
        'corners': corners,
        'detected_aruco_ids': sorted(list(vision_mgr.last_detected_markers.keys()))
    })

@app.route('/api/vision/detection-config', methods=['POST'])
def vision_detection_config():
    data = request.get_json(silent=True) or {}
    raw_frame = camera_mgr.get_raw_frame()
    cfg = vision_mgr.update_detection_config(data, raw_frame=raw_frame)
    return jsonify({
        'success': True,
        'config': cfg,
        'corners': vision_mgr.corners_norm,
        'detected_aruco_ids': sorted(list(vision_mgr.last_detected_markers.keys())),
        'status': vision_mgr.get_status()
    })

@app.route('/api/vision/detection-config/reset', methods=['POST'])
def vision_detection_config_reset():
    raw_frame = camera_mgr.get_raw_frame()
    cfg = vision_mgr.reset_detection_config_to_defaults(raw_frame=raw_frame)
    return jsonify({
        'success': True,
        'config': cfg,
        'corners': vision_mgr.corners_norm,
        'detected_aruco_ids': sorted(list(vision_mgr.last_detected_markers.keys())),
        'status': vision_mgr.get_status()
    })

@app.route('/api/vision/corners', methods=['POST'])
def vision_corners():
    data = request.get_json(silent=True) or {}
    corners = data.get('corners')
    if corners and len(corners) == 4:
        vision_mgr.set_corners(corners)
        vision_mgr.refit_grid(camera_mgr.get_raw_frame())
        return jsonify({'success': True, 'corners': vision_mgr.corners_norm})
    return jsonify({'success': False, 'error': 'Format de coins invalide'}), 400

@app.route('/api/vision/rotate-board', methods=['POST'])
def vision_rotate_board():
    new_corners = vision_mgr.rotate_board_90()
    return jsonify({'success': True, 'corners': new_corners})

@app.route('/api/vision/overlay-settings', methods=['POST'])
def vision_overlay_settings():
    data = request.get_json(silent=True) or {}
    if 'show_ar_grid' in data:
        vision_mgr.show_ar_grid = bool(data['show_ar_grid'])
    if 'show_ar_labels' in data:
        vision_mgr.show_ar_labels = bool(data['show_ar_labels'])
    if 'show_ar_arrow' in data:
        vision_mgr.show_ar_arrow = bool(data['show_ar_arrow'])
    if 'show_ar_vision_debug' in data:
        vision_mgr.show_ar_vision_debug = bool(data['show_ar_vision_debug'])
    if 'show_ar_pieces' in data:
        vision_mgr.show_ar_pieces = bool(data['show_ar_pieces'])
    if 'show_warped_grid' in data:
        vision_mgr.show_warped_grid = bool(data['show_warped_grid'])
    if 'show_warped_arrow' in data:
        vision_mgr.show_warped_arrow = bool(data['show_warped_arrow'])
    if 'show_aruco_debug' in data:
        vision_mgr.show_aruco_debug = bool(data['show_aruco_debug'])
    vision_mgr.save_calibration()
    return jsonify({'success': True, 'status': vision_mgr.get_status()})

@app.route('/api/vision/auto-calibrate-pieces', methods=['POST'])
def vision_auto_calibrate_pieces():
    warped = vision_mgr.get_latest_warped_frame()
    if warped is None:
        return jsonify({'success': False, 'error': 'Aucune image webcam disponible'}), 400
    from piece_auto_calibrator import auto_calibrator
    cur_padded = getattr(vision_mgr, 'get_latest_warped_padded', lambda: None)()
    p_center = getattr(vision_mgr, 'get_perspective_center', lambda: None)()
    ok, msg, summary = auto_calibrator.calibrate_from_starting_board(warped, warped_padded=cur_padded, center_xy=p_center)
    return jsonify({'success': ok, 'message': msg, 'summary': summary})

@app.route('/api/vision/warped_stream')
def vision_warped_stream():
    def generate():
        while camera_mgr.is_running:
            jpeg = vision_mgr.get_warped_jpeg()
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Routes Jeu d'Échecs & Suivi des Coups (Slice 2 & 3) ---
@app.route('/api/game/status')
def game_status():
    return jsonify({
        'success': True,
        'game': game_mgr.get_status(),
        'engine': {
            'stockfish_available': engine_mgr.stockfish_available,
            'bot_elo': engine_mgr.bot_elo,
            'game_mode': engine_mgr.game_mode,
            'human_color': 'white' if engine_mgr.human_color == 1 else 'black'
        }
    })

@app.route('/api/game/new', methods=['POST'])
def game_new():
    data = request.get_json(silent=True) or {}
    fen = data.get('fen')
    status = game_mgr.start_new_game(fen=fen)
    return jsonify({'success': True, 'game': status})

@app.route('/api/game/confirm-setup', methods=['POST'])
def game_confirm_setup():
    ok, msg = game_mgr.confirm_setup()
    if ok:
        warped = vision_mgr.get_latest_warped_frame()
        if warped is not None:
            ok_ds, msg_ds, count_ds = dataset_collector.save_labeled_board(warped, game_mgr.board)
            if ok_ds:
                log(f"📸 Dataset : {count_ds} cases capturées et enregistrées")
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()})

@app.route('/api/game/resync-reference', methods=['POST'])
def game_resync_reference():
    ok, msg = game_mgr.resync_reference()
    return jsonify({'success': ok, 'message': msg})

@app.route('/api/game/manual-move', methods=['POST'])
def game_manual_move():
    data = request.get_json(silent=True) or {}
    uci_str = data.get('uci', '').strip()
    ok, msg = game_mgr.play_manual_move(uci_str)
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()})

@app.route('/api/game/set-position', methods=['POST'])
def game_set_position():
    data = request.get_json(silent=True) or {}
    ok, msg = game_mgr.set_position(data.get('fen', '').strip())
    if ok:
        warped = vision_mgr.get_latest_warped_frame()
        if warped is not None:
            ok_ds, msg_ds, count_ds = dataset_collector.save_labeled_board(warped, game_mgr.board)
            if ok_ds:
                log(f"📸 Dataset : {count_ds} cases capturées et enregistrées")
                msg = f"{msg} (📸 {count_ds} cases capturées pour le dataset)"
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()}), (200 if ok else 400)

@app.route('/api/game/undo', methods=['POST'])
def game_undo():
    ok = game_mgr.undo_last_move()
    return jsonify({'success': ok, 'game': game_mgr.get_status()})

@app.route('/api/game/hint', methods=['POST'])
def game_hint():
    hint_res = engine_mgr.get_hint(game_mgr.board)
    if hint_res:
        m = hint_res["move"]
        vision_mgr.ar_arrow = {
            "from_sq": m.from_square,
            "to_sq": m.to_square,
            "is_capture": game_mgr.board.is_capture(m),
            "color": (255, 200, 0),  # Flèche Cyan/Bleue pour l'indice
            "label": f"Conseil: {hint_res['san']}",
            "kind": "hint"
        }
        return jsonify({'success': True, 'hint': hint_res})
    return jsonify({'success': False, 'error': 'Aucun conseil disponible'}), 400

@app.route('/api/game/saved', methods=['GET'])
def game_saved_list():
    games = game_mgr.list_saved_games()
    return jsonify({'success': True, 'games': games})

@app.route('/api/game/saved/load', methods=['POST'])
def game_saved_load():
    data = request.get_json(silent=True) or {}
    game_id = data.get('game_id')
    if not game_id:
        return jsonify({'success': False, 'error': 'ID de partie manquant'}), 400
    ok, msg = game_mgr.load_saved_game_by_id(game_id)
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()}), (200 if ok else 400)

@app.route('/api/game/saved/delete', methods=['POST'])
def game_saved_delete():
    data = request.get_json(silent=True) or {}
    game_id = data.get('game_id')
    if not game_id:
        return jsonify({'success': False, 'error': 'ID de partie manquant'}), 400
    ok, msg = game_mgr.delete_saved_game_by_id(game_id)
    return jsonify({'success': ok, 'message': msg})

@app.route('/api/game/saved/export-pgn', methods=['GET'])
def game_saved_export_pgn():
    game_id = request.args.get('game_id')
    pgn_text = game_mgr.export_pgn(game_id)
    if not pgn_text:
        return jsonify({'success': False, 'error': 'Partie introuvable ou vide'}), 404
    return Response(pgn_text, mimetype='text/plain', headers={'Content-Disposition': f'attachment; filename={game_id or "game"}.pgn'})

@app.route('/api/game/saved/save-checkpoint', methods=['POST'])
def game_saved_checkpoint():
    game_mgr._save_game()
    return jsonify({"success": True, "message": "Partie sauvegardée dans l'archive avec succès", "game_id": game_mgr.current_game_id})

@app.route('/api/engine/settings', methods=['POST'])
def engine_settings():
    data = request.get_json(silent=True) or {}
    if 'bot_elo' in data:
        engine_mgr.bot_elo = int(data['bot_elo'])
    if 'game_mode' in data:
        engine_mgr.game_mode = data['game_mode']
    if 'human_color' in data:
        engine_mgr.human_color = 1 if data['human_color'] == 'white' else 0
    return jsonify({'success': True, 'engine': {
        'bot_elo': engine_mgr.bot_elo,
        'game_mode': engine_mgr.game_mode,
        'stockfish_available': engine_mgr.stockfish_available
    }})

# --- Routes Robot SO-101 Bridge (Slice 4) ---
@app.route('/api/robot/grid-coordinates')
def robot_grid_coordinates():
    grid = {}
    import chess
    for sq in chess.SQUARES:
        grid[chess.square_name(sq)] = robot_bridge.get_square_spatial_info(sq, vision_mgr)
    return jsonify({'success': True, 'coordinates': grid})

@app.route('/api/robot/plan-move', methods=['POST'])
def robot_plan_move():
    data = request.get_json(silent=True) or {}
    uci = data.get('uci')
    import chess
    try:
        move = chess.Move.from_uci(uci) if uci else (game_mgr.pending_bot_move['move'] if game_mgr.pending_bot_move else None)
        if not move:
            return jsonify({'success': False, 'error': 'Aucun coup spécifié ou en attente'}), 400
        plan = robot_bridge.build_robot_action_plan(game_mgr.board, move, vision_mgr)
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# --- Routes Machine Learning Milieu de Partie (Slice 5) ---
@app.route('/api/ml/status')
def ml_status():
    stats = dataset_collector.get_dataset_stats()
    return jsonify({
        'success': True,
        'dataset': stats,
        'onnx_ready': onnx_classifier.is_ready()
    })

@app.route('/api/ml/capture-sample', methods=['POST'])
def ml_capture_sample():
    warped = vision_mgr.get_latest_warped_frame()
    ok, msg, count = dataset_collector.save_labeled_board(warped, game_mgr.board)
    return jsonify({'success': ok, 'message': msg, 'saved_count': count})

@app.route('/api/ml/scan-position', methods=['POST'])
def ml_scan_position():
    data = request.get_json(silent=True) or {}
    turn = 1 if data.get('turn', 'white') == 'white' else 0
    warped = vision_mgr.get_latest_warped_frame()
    if warped is None:
        return jsonify({'success': False, 'error': "Aucune image reçue"}), 400

    res, err = onnx_classifier.predict_board_fen(warped, active_turn=turn)
    if err:
        return jsonify({'success': False, 'error': err}), 400

    # Mettre à jour le jeu avec la nouvelle position
    game_mgr.start_new_game(fen=res['fen'])
    game_mgr.confirm_setup()
    return jsonify({'success': True, 'result': res, 'game': game_mgr.get_status()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
