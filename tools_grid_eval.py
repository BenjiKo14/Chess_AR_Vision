"""Évalue l'alignement du quadrillage (avec / sans Hough) sur un snapshot et génère des images de comparaison."""
import sys
import cv2
import numpy as np
from chess_vision import ChessVisionManager, WARP_SIZE

path = sys.argv[1] if len(sys.argv) > 1 else 'snapshots/snapshot_20261031_005135.jpg'
raw = cv2.imread(path)
vm = ChessVisionManager()
ok, msg, corners = vm.auto_detect_board(raw)
print(msg)

warped = vm.warp_frame(raw)
gray = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (3, 3), 0)
gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def score(grid):
    """Gradient moyen échantillonné le long des 14 lignes internes (plus haut = mieux aligné)."""
    vals = []
    for k in range(1, 8):
        for t in np.linspace(5, WARP_SIZE - 5, 200):
            # verticale k
            xt, xb = grid['v'][k]
            x = xt + (xb - xt) * t / WARP_SIZE
            vals.append(gx[int(t), int(round(np.clip(x, 0, WARP_SIZE - 1)))])
            yl, yr = grid['h'][k]
            y = yl + (yr - yl) * t / WARP_SIZE
            vals.append(gy[int(round(np.clip(y, 0, WARP_SIZE - 1))), int(t)])
    return float(np.mean(vals))


uni = vm._uniform_grid()
ref = vm.refined_grid
print(f"Score SANS Hough : {score(uni):.1f}")
print(f"Score AVEC Hough : {score(ref):.1f}  ({vm.refined_grid_count}/14 lignes)")
for k in range(1, 8):
    print(f"  v{k}: {ref['v'][k][0]-64*k:+.1f}/{ref['v'][k][1]-64*k:+.1f}   h{k}: {ref['h'][k][0]-64*k:+.1f}/{ref['h'][k][1]-64*k:+.1f}")

for flag, name in ((False, 'sans'), (True, 'avec')):
    vm.detection_config['use_hough_refine'] = flag
    w = warped.copy()
    vm._draw_warped_grid_overlay(w)
    cv2.imwrite(f'snapshots/grid_warped_{name}_hough.jpg', w)
    cam = raw.copy()
    vm._draw_ar_elements(cam, raw.shape[1], raw.shape[0])
    cv2.imwrite(f'snapshots/grid_camera_{name}_hough.jpg', cam)
