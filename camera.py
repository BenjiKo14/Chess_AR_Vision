import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

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

class CameraManager:
    """Gestionnaire d'acquisition webcam en thread dédié."""
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or safe_log_print
        self.camera_index = None
        self.cap = None
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Frames en mémoire
        self._latest_raw_frame = None       # Image OpenCV propre (sans retouche ni overlay)
        self._latest_display_jpeg = None    # Bytes JPEG avec overlays AR pour stream principal

        self.width = 640
        self.height = 480
        self.fps = 0.0

        # Réglages caméra
        self.rotate_180 = False
        self.flip_h = False
        self.show_crosshair = False
        self.active_filters = []  # Liste cumulable : ["clahe", "binary", "adaptive", "sharpen"]

        # Hook processeur d'image (ex: ChessVisionManager.process_frame)
        self.frame_processor = None
        self.is_static_image = False

    def load_static_image(self, file_path_or_bytes):
        """Charge une image fixe (photo) à la place du flux webcam."""
        with self._lock:
            if self.is_running and not self.is_static_image:
                self._stop_unlocked()

            if isinstance(file_path_or_bytes, (str, Path)):
                img = cv2.imread(str(file_path_or_bytes))
            else:
                nparr = np.frombuffer(file_path_or_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return False, "Impossible de charger l'image"

            h, w = img.shape[:2]
            self.width = w
            self.height = h
            self.camera_index = "photo"
            self.is_running = True
            self.is_static_image = True
            self._latest_raw_frame = img.copy()

            # Lancer une boucle légère pour continuer à servir le flux et l'AR interactive
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._static_loop, daemon=True)
            self._thread.start()
            self.log(f"[OK] Photo chargee comme source video ({w}x{h})")
            return True, None

    def _apply_preview_filter(self, img):
        if not self.active_filters:
            return img

        res = img.copy()
        is_gray = False

        for f in self.active_filters:
            if f == "clahe":
                if not is_gray:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    is_gray = True
                cl = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                res = cl.apply(res)
            elif f == "binary":
                if not is_gray:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    is_gray = True
                _, res = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY)
            elif f == "otsu":
                if not is_gray:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    is_gray = True
                _, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif f == "adaptive":
                if not is_gray:
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                    is_gray = True
                res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 4)
            elif f == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                res = cv2.filter2D(res, -1, kernel)

        if is_gray:
            return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        return res

    def _static_loop(self):
        """Sert l'image fixe en continu pour permettre l'interaction AR sans webcam."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._latest_raw_frame is None:
                    break
                frame = self._latest_raw_frame.copy()

            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self.flip_h:
                frame = cv2.flip(frame, 1)

            display_frame = frame.copy()
            if self.frame_processor:
                try:
                    display_frame = self.frame_processor(display_frame)
                except Exception as e:
                    pass

            display_frame = self._apply_preview_filter(display_frame)

            ok, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._lock:
                    self._latest_display_jpeg = buf.tobytes()

            time.sleep(0.04)

        with self._lock:
            self._stop_unlocked()

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

    def list_cameras(self, max_index=4):
        """Détecte les webcams branchées sous Windows/Linux sans bloquer."""
        cameras = []
        is_windows = (os.name == 'nt')
        backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

        for idx in range(max_index):
            if self.is_running and self.camera_index == idx:
                cameras.append({
                    'index': idx,
                    'label': f"Caméra {idx} ({self.width}x{self.height}) [Active]",
                    'active': True
                })
                continue

            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cameras.append({
                            'index': idx,
                            'label': f"Caméra {idx} ({w}x{h})",
                            'active': False
                        })
                    cap.release()
            except Exception:
                pass

        return cameras

    def start(self, index=0, width=640, height=480):
        """Démarre le flux sur la caméra demandée."""
        with self._lock:
            if self.is_running:
                if self.camera_index == index:
                    return True, None
                self._stop_unlocked()

            self.log(f"📷 Ouverture de la caméra {index}...")
            is_windows = (os.name == 'nt')
            backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

            cap = cv2.VideoCapture(int(index), backend)
            if not cap.isOpened() and is_windows:
                cap = cv2.VideoCapture(int(index))

            if not cap.isOpened():
                self.log(f"❌ Impossible d'ouvrir la caméra {index}")
                return False, f"Impossible d'ouvrir la caméra {index}"

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                self.log(f"❌ La caméra {index} ne renvoie pas d'image")
                return False, f"La caméra {index} ne renvoie pas d'image"

            self.cap = cap
            self.camera_index = index
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            self.is_running = True
            self._stop_event.clear()

            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            self.log(f"✅ Caméra {index} active ({self.width}x{self.height})")
            return True, None

    def _capture_loop(self):
        """Boucle d'acquisition continue dans un thread dédié."""
        fail_count = 0
        last_time = time.time()
        frame_count = 0

        while not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                fail_count += 1
                if fail_count > 25:
                    self.log(f"⚠️ Perte du signal vidéo caméra {self.camera_index}")
                    break
                time.sleep(0.04)
                continue

            fail_count = 0

            # 1. Orientation physique de la caméra
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self.flip_h:
                frame = cv2.flip(frame, 1)

            # 2. Sauvegarde de la frame propre
            with self._lock:
                self._latest_raw_frame = frame.copy()

            # 3. Traitement AR via frame_processor
            display_frame = frame.copy()
            if self.frame_processor:
                try:
                    display_frame = self.frame_processor(display_frame)
                except Exception as e:
                    self.log(f"Erreur frame_processor: {e}")

            if self.show_crosshair:
                h, w = display_frame.shape[:2]
                cv2.circle(display_frame, (w // 2, h // 2), 15, (0, 255, 200), 1, cv2.LINE_AA)

            display_frame = self._apply_preview_filter(display_frame)

            # 4. Encodage JPEG pour streaming web
            ok, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._lock:
                    self._latest_display_jpeg = buf.tobytes()

            # Calcul FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            time.sleep(0.015)

        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self):
        self._stop_event.set()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_running = False
        self.camera_index = None
        self._latest_display_jpeg = None
        self.fps = 0.0
        self.is_static_image = False

    def stop(self):
        with self._lock:
            self.log("⏹ Arrêt de la caméra...")
            self._stop_unlocked()
            return True

    def get_raw_frame(self):
        """Retourne la dernière image brute reçue (thread-safe)."""
        with self._lock:
            return self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None

    def get_jpeg_frame(self):
        """Retourne les bytes JPEG pour la route MJPEG."""
        with self._lock:
            return self._latest_display_jpeg

    def update_settings(self, rotate_180=None, flip_h=None, show_crosshair=None, active_filters=None, toggle_flip_h=False, toggle_rotate_180=False):
        with self._lock:
            if toggle_flip_h:
                self.flip_h = not self.flip_h
            elif flip_h is not None:
                self.flip_h = bool(flip_h)

            if toggle_rotate_180:
                self.rotate_180 = not self.rotate_180
            elif rotate_180 is not None:
                self.rotate_180 = bool(rotate_180)

            if show_crosshair is not None:
                self.show_crosshair = bool(show_crosshair)
            if active_filters is not None:
                self.active_filters = list(active_filters)
            return self.get_status()

    def take_snapshot(self, save_dir="snapshots"):
        with self._lock:
            if not self.is_running or self._latest_raw_frame is None:
                return False, "Caméra non active ou pas d'image", None
            frame = self._latest_raw_frame.copy()

        try:
            dir_path = Path(save_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = dir_path / filename
            cv2.imwrite(str(filepath), frame)
            return True, filename, str(filepath)
        except Exception as e:
            return False, str(e), None

    def get_status(self):
        return {
            'connected': self.is_running,
            'index': self.camera_index,
            'width': self.width,
            'height': self.height,
            'fps': round(self.fps, 1),
            'settings': {
                'rotate_180': self.rotate_180,
                'flip_h': self.flip_h,
                'show_crosshair': self.show_crosshair,
                'active_filters': self.active_filters
            }
        }
