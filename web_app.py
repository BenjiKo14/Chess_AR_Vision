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
    status = camera_mgr.update_settings(
        rotate_180=data.get('rotate_180'),
        flip_h=data.get('flip_h'),
        show_crosshair=data.get('show_crosshair')
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
    if ok:
        return jsonify({'success': True, 'status': camera_mgr.get_status()})
    return jsonify({'success': False, 'error': err}), 400

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory('snapshots', filename)

# --- Routes Vision & Calibration (Slice 1 & 2) ---
@app.route('/api/vision/status')
def vision_status():
    return jsonify({'success': True, **vision_mgr.get_status()})

@app.route('/api/vision/auto-calibrate', methods=['POST'])
def vision_auto_calibrate():
    raw_frame = camera_mgr.get_raw_frame()
    if raw_frame is None:
        return jsonify({'success': False, 'error': 'Aucune image reçue de la caméra'}), 400
    ok, msg, corners = vision_mgr.auto_detect_board(raw_frame)
    return jsonify({'success': ok, 'message': msg, 'corners': corners})

@app.route('/api/vision/corners', methods=['POST'])
def vision_corners():
    data = request.get_json(silent=True) or {}
    corners = data.get('corners')
    if corners and len(corners) == 4:
        vision_mgr.set_corners(corners)
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
    if 'show_warped_grid' in data:
        vision_mgr.show_warped_grid = bool(data['show_warped_grid'])
    return jsonify({'success': True, 'status': vision_mgr.get_status()})

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
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()})

@app.route('/api/game/resync-reference', methods=['POST'])
def game_resync_reference():
    ok = game_mgr.resync_reference()
    return jsonify({'success': ok, 'message': "Image de référence synchronisée"})

@app.route('/api/game/manual-move', methods=['POST'])
def game_manual_move():
    data = request.get_json(silent=True) or {}
    uci_str = data.get('uci', '').strip()
    ok, msg = game_mgr.play_manual_move(uci_str)
    return jsonify({'success': ok, 'message': msg, 'game': game_mgr.get_status()})

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
            "label": f"Conseil: {hint_res['san']}"
        }
        return jsonify({'success': True, 'hint': hint_res})
    return jsonify({'success': False, 'error': 'Aucun conseil disponible'}), 400

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
