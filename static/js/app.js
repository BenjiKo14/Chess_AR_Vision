// État global frontend
let cameraConnected = false;
let editingCorners = false;
let activeDragCorner = null;
let currentCornersNorm = [[0.2, 0.15], [0.8, 0.15], [0.85, 0.85], [0.15, 0.85]];

// Mode 4 clics guidés (dans n'importe quel ordre !)
let guide4PointsMode = false;
let clickedPointsNorm = []; // Liste des clics bruts [ [x, y], ... ]
let lastMousePos = null;    // Pour effet réticule / curseur interactif

const PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
};

document.addEventListener('DOMContentLoaded', () => {
    initDigitalBoard();
    initEventListeners();
    detectCameras();
    fetchVisionStatus();
    fetchGameStatus();
    setInterval(pollGameStatus, 1000);
    setInterval(pollLogs, 2000);
});

function initDigitalBoard() {
    const boardEl = document.getElementById('digital-chess-board');
    boardEl.innerHTML = '';
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const sqEl = document.createElement('div');
            const isLight = (r + c) % 2 === 0;
            const sqIndex = (7 - r) * 8 + c;
            sqEl.className = `chess-sq ${isLight ? 'light' : 'dark'}`;
            sqEl.id = `sq-${sqIndex}`;
            sqEl.dataset.sq = sqIndex;
            boardEl.appendChild(sqEl);
        }
    }
}

function updateDigitalBoard(fen, lastMoveUci) {
    if (!fen) return;
    const parts = fen.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
        const rowStr = parts[r];
        let col = 0;
        for (let ch of rowStr) {
            if (ch >= '1' && ch <= '8') {
                const emptyCount = parseInt(ch);
                for (let k = 0; k < emptyCount; k++) {
                    const sqIdx = (7 - r) * 8 + col;
                    const sqEl = document.getElementById(`sq-${sqIdx}`);
                    if (sqEl) sqEl.innerText = '';
                    col++;
                }
            } else {
                const sqIdx = (7 - r) * 8 + col;
                const sqEl = document.getElementById(`sq-${sqIdx}`);
                if (sqEl) sqEl.innerText = PIECE_UNICODE[ch] || ch;
                col++;
            }
        }
    }

    // Effacer highlights
    document.querySelectorAll('.chess-sq').forEach(el => {
        el.classList.remove('highlight-from', 'highlight-to');
    });

    if (lastMoveUci && lastMoveUci.length >= 4) {
        const fCol = lastMoveUci.charCodeAt(0) - 97;
        const fRow = parseInt(lastMoveUci[1]) - 1;
        const tCol = lastMoveUci.charCodeAt(2) - 97;
        const tRow = parseInt(lastMoveUci[3]) - 1;

        const fromIdx = fRow * 8 + fCol;
        const toIdx = tRow * 8 + tCol;

        const fromEl = document.getElementById(`sq-${fromIdx}`);
        const toEl = document.getElementById(`sq-${toIdx}`);
        if (fromEl) fromEl.classList.add('highlight-from');
        if (toEl) toEl.classList.add('highlight-to');
    }
}

function initEventListeners() {
    // Caméra
    document.getElementById('btn-detect-cam').addEventListener('click', detectCameras);
    document.getElementById('btn-toggle-cam').addEventListener('click', toggleCamera);
    document.getElementById('input-upload-photo').addEventListener('change', uploadPhoto);
    document.getElementById('btn-rotate-cam').addEventListener('click', () => {
        postJSON('/api/camera/settings', { rotate_180: true });
    });
    document.getElementById('btn-flip-cam').addEventListener('click', () => {
        postJSON('/api/camera/settings', { flip_h: true });
    });
    document.getElementById('btn-snapshot').addEventListener('click', () => {
        postJSON('/api/camera/snapshot', {});
    });

    // Vision & Calibration
    document.getElementById('btn-auto-calib').addEventListener('click', autoCalibrate);
    document.getElementById('btn-guide-4pts').addEventListener('click', start4PointsGuide);
    document.getElementById('btn-rotate-board').addEventListener('click', () => {
        postJSON('/api/vision/rotate-board', {}).then(res => {
            if (res.corners) {
                currentCornersNorm = res.corners;
                drawCornersOverlay();
            }
        });
    });
    document.getElementById('btn-edit-corners').addEventListener('click', toggleEditCorners);

    document.getElementById('btn-toggle-grid').addEventListener('click', function() {
        this.classList.toggle('active');
        postJSON('/api/vision/overlay-settings', { show_ar_grid: this.classList.contains('active') });
    });
    document.getElementById('btn-toggle-labels').addEventListener('click', function() {
        this.classList.toggle('active');
        postJSON('/api/vision/overlay-settings', { show_ar_labels: this.classList.contains('active') });
    });

    // Contrôles Partie
    document.getElementById('btn-new-game').addEventListener('click', () => {
        postJSON('/api/game/new', {}).then(pollGameStatus);
    });
    document.getElementById('btn-confirm-setup').addEventListener('click', () => {
        postJSON('/api/game/confirm-setup', {}).then(pollGameStatus);
    });
    document.getElementById('btn-resync').addEventListener('click', () => {
        postJSON('/api/game/resync-reference', {}).then(pollGameStatus);
    });
    document.getElementById('btn-undo').addEventListener('click', () => {
        postJSON('/api/game/undo', {}).then(pollGameStatus);
    });
    document.getElementById('btn-hint').addEventListener('click', () => {
        postJSON('/api/game/hint', {}).then(res => {
            if (res.hint) {
                document.getElementById('banner-action').innerText = `💡 Conseil : ${res.hint.san} (${res.hint.uci})`;
            }
        });
    });

    // Réglages Moteur
    document.getElementById('select-bot-elo').addEventListener('change', function() {
        postJSON('/api/engine/settings', { bot_elo: parseInt(this.value) });
    });
    document.getElementById('select-game-mode').addEventListener('change', function() {
        postJSON('/api/engine/settings', { game_mode: this.value });
    });

    // Outils Milieu de partie et Robot
    document.getElementById('btn-scan-ml').addEventListener('click', () => {
        postJSON('/api/ml/scan-position', { turn: 'white' }).then(res => {
            if (res.success) {
                alert(`✅ Position détectée par IA ONNX : ${res.result.fen}`);
                pollGameStatus();
            } else {
                alert(`Erreur scan : ${res.error}`);
            }
        });
    });

    document.getElementById('btn-capture-ml').addEventListener('click', () => {
        postJSON('/api/ml/capture-sample', {}).then(res => {
            alert(res.message);
        });
    });

    document.getElementById('btn-plan-robot').addEventListener('click', () => {
        postJSON('/api/robot/plan-move', {}).then(res => {
            if (res.success) {
                const planStr = res.plan.actions.map((a, i) => `${i+1}. ${a.action} (${a.piece || ''})`).join('\n');
                alert(`🦾 Plan Robot pour ${res.plan.move_uci} :\n\n${planStr}`);
            } else {
                alert(`Erreur robot : ${res.error}`);
            }
        });
    });

    // Canvas interactif pour drag & drop des 4 coins
    setupCanvasInteraction();
}

function detectCameras() {
    fetch('/api/camera/list')
        .then(r => r.json())
        .then(data => {
            const sel = document.getElementById('camera-select');
            sel.innerHTML = '';
            if (data.cameras && data.cameras.length > 0) {
                data.cameras.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.index;
                    opt.text = c.label;
                    if (c.active) opt.selected = true;
                    sel.appendChild(opt);
                });
            } else {
                sel.innerHTML = '<option value="0">Caméra 0 (Par défaut)</option>';
            }
        });
}

function uploadPhoto(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('photo', file);

    const banner = document.getElementById('banner-action');
    banner.innerText = "⏳ Chargement de la photo...";

    fetch('/api/camera/upload-photo', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(res => {
        if (res.success) {
            cameraConnected = true;
            const btn = document.getElementById('btn-toggle-cam');
            const imgStream = document.getElementById('cam-stream');
            const warpedStream = document.getElementById('warped-stream');
            const placeholder = document.getElementById('cam-placeholder');
            const warpedPlaceholder = document.getElementById('warped-placeholder');
            const badge = document.getElementById('cam-status-badge');

            btn.innerText = '⏹ Fermer Photo';
            btn.className = 'btn btn-secondary';
            badge.innerText = 'Photo Active';
            badge.className = 'badge badge-online';

            imgStream.src = `/api/camera/stream?t=${Date.now()}`;
            imgStream.style.display = 'block';
            placeholder.style.display = 'none';

            warpedStream.src = `/api/vision/warped_stream?t=${Date.now()}`;
            warpedStream.style.display = 'block';
            warpedPlaceholder.style.display = 'none';

            banner.innerText = `🖼️ Photo chargée (${file.name}). Vous pouvez maintenant ajuster les coins ou scanner la position !`;
            setTimeout(resizeCanvas, 500);
        } else {
            alert('Erreur chargement photo: ' + res.error);
        }
    });
}

function toggleCamera() {
    const btn = document.getElementById('btn-toggle-cam');
    const imgStream = document.getElementById('cam-stream');
    const warpedStream = document.getElementById('warped-stream');
    const placeholder = document.getElementById('cam-placeholder');
    const warpedPlaceholder = document.getElementById('warped-placeholder');
    const badge = document.getElementById('cam-status-badge');

    if (!cameraConnected) {
        const idx = document.getElementById('camera-select').value;
        postJSON('/api/camera/connect', { index: parseInt(idx) })
            .then(res => {
                if (res.success) {
                    cameraConnected = true;
                    btn.innerText = '⏹ Arrêter Caméra';
                    btn.className = 'btn btn-secondary';
                    badge.innerText = 'Caméra En Direct';
                    badge.className = 'badge badge-online';

                    imgStream.src = `/api/camera/stream?t=${Date.now()}`;
                    imgStream.style.display = 'block';
                    placeholder.style.display = 'none';

                    warpedStream.src = `/api/vision/warped_stream?t=${Date.now()}`;
                    warpedStream.style.display = 'block';
                    warpedPlaceholder.style.display = 'none';

                    setTimeout(resizeCanvas, 500);
                } else {
                    alert('Erreur activation caméra: ' + res.error);
                }
            });
    } else {
        imgStream.src = '';
        imgStream.style.display = 'none';
        warpedStream.src = '';
        warpedStream.style.display = 'none';
        placeholder.style.display = 'block';
        warpedPlaceholder.style.display = 'block';

        postJSON('/api/camera/disconnect', {})
            .then(() => {
                cameraConnected = false;
                btn.innerText = '▶ Activer Caméra';
                btn.className = 'btn btn-primary';
                badge.innerText = 'Caméra Inactive';
                badge.className = 'badge badge-offline';
            });
    }
}

function autoCalibrate() {
    const btn = document.getElementById('btn-auto-calib');
    const banner = document.getElementById('banner-action');
    const originalText = '✨ Détection Auto (ArUco / Grille)';
    
    btn.disabled = true;
    btn.innerText = '⏳ Analyse ArUco & Grille...';
    banner.innerText = '⏳ Recherche des marqueurs ArUco aux 4 coins...';

    postJSON('/api/vision/auto-calibrate', {})
        .then(res => {
            btn.disabled = false;
            btn.innerText = originalText;
            if (res.success && res.corners) {
                currentCornersNorm = res.corners;
                drawOverlay();
                banner.innerText = `✅ Succès : ${res.message}`;
            } else {
                banner.innerText = `⚠️ ${res.message || "Aucun marqueur détecté"}. Vous pouvez aussi utiliser « 🎯 Pointer 4 Clics ».`;
            }
        })
        .catch(err => {
            btn.disabled = false;
            btn.innerText = originalText;
            banner.innerText = `❌ Erreur de communication serveur : ${err}`;
        });
}

function fetchVisionStatus() {
    fetch('/api/vision/status')
        .then(r => r.json())
        .then(data => {
            if (data.corners_norm) {
                currentCornersNorm = data.corners_norm;
                drawCornersOverlay();
            }
        });
}

function fetchGameStatus() {
    fetch('/api/game/status')
        .then(r => r.json())
        .then(updateGameUI);
}

function pollGameStatus() {
    if (cameraConnected) {
        fetch('/api/camera/status').then(r => r.json()).then(data => {
            document.getElementById('fps-counter').innerText = `${data.fps} FPS`;
        });
    }
    fetchGameStatus();
}

function updateGameUI(data) {
    if (!data || !data.game) return;
    const g = data.game;
    updateDigitalBoard(g.fen, g.last_move);

    const turnBadge = document.getElementById('turn-badge');
    const banner = document.getElementById('banner-action');

    turnBadge.innerText = g.state;
    if (g.state === 'WAIT_HUMAN_MOVE') {
        turnBadge.className = 'badge badge-online';
        banner.innerText = `🟢 À votre tour de jouer (${g.turn === 'white' ? 'Blancs' : 'Noirs'}). Déplacez une pièce sur le plateau.`;
    } else if (g.state === 'WAIT_BOT_EXECUTION' && g.pending_bot_move) {
        turnBadge.className = 'badge badge-offline';
        banner.innerText = `🤖 Déplacez la pièce du Bot : ${g.pending_bot_move.san} (${g.pending_bot_move.uci}) sur votre échiquier !`;
    } else if (g.state === 'SETUP_CHECK') {
        banner.innerText = `♟️ Placez les 32 pièces sur les rangées indiquées en AR, puis cliquez sur « ✅ Valider Pièces Initiales ».`;
    }

    // Historique
    if (g.history && g.history.length > 0) {
        const historyBox = document.getElementById('move-history-box');
        historyBox.innerHTML = g.history.map(m => `<span>#${m.num} <b>${m.san}</b> (${m.player})</span>`).join(' • ');
    }
}

function pollLogs() {
    fetch('/api/logs')
        .then(r => r.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                const box = document.getElementById('logs-box');
                box.innerHTML = data.logs.slice(-6).join('<br>');
                box.scrollTop = box.scrollHeight;
            }
        });
}

function getImageRenderInfo() {
    const img = document.getElementById('cam-stream');
    const container = document.getElementById('viewport-box');
    if (!img || !img.naturalWidth || !container) {
        return null;
    }

    const containerW = container.clientWidth;
    const containerH = container.clientHeight;
    const imgNatW = img.naturalWidth;
    const imgNatH = img.naturalHeight;

    const scale = Math.min(containerW / imgNatW, containerH / imgNatH);
    const renderedW = imgNatW * scale;
    const renderedH = imgNatH * scale;

    const offsetX = (containerW - renderedW) / 2;
    const offsetY = (containerH - renderedH) / 2;

    return {
        containerW, containerH,
        renderedW, renderedH,
        offsetX, offsetY,
        scale,
        imgNatW, imgNatH
    };
}

function start4PointsGuide() {
    guide4PointsMode = true;
    clickedPointsNorm = [];
    editingCorners = false;

    const canvas = document.getElementById('interactive-canvas');
    canvas.classList.add('interactive');
    resizeCanvas();

    const banner = document.getElementById('banner-action');
    banner.innerText = `🎯 Mode 4 Clics : Cliquez sur n'importe lequel des 4 coins (0/4)`;
    document.getElementById('btn-guide-4pts').classList.add('active');
    drawOverlay();
}

function resizeCanvas() {
    const container = document.getElementById('viewport-box');
    const canvas = document.getElementById('interactive-canvas');
    if (container && canvas) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        drawOverlay();
    }
}

window.addEventListener('resize', resizeCanvas);

function clearCanvas() {
    const canvas = document.getElementById('interactive-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function sortCornersGeometrically(pts) {
    if (pts.length !== 4) return pts;
    // Tri géométrique robuste : 0=Haut-Gauche (A8), 1=Haut-Droite (H8), 2=Bas-Droite (H1), 3=Bas-Gauche (A1)
    const sums = pts.map(p => p[0] + p[1]);
    const diffs = pts.map(p => p[0] - p[1]);

    const tlIdx = sums.indexOf(Math.min(...sums));
    const brIdx = sums.indexOf(Math.max(...sums));
    const trIdx = diffs.indexOf(Math.max(...diffs));
    const blIdx = diffs.indexOf(Math.min(...diffs));

    return [pts[tlIdx], pts[trIdx], pts[brIdx], pts[blIdx]];
}

function drawOverlay() {
    const canvas = document.getElementById('interactive-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const info = getImageRenderInfo();
    if (!info) return;

    // 1. Rendu pendant le mode 4 Clics
    if (guide4PointsMode) {
        clickedPointsNorm.forEach((pt, idx) => {
            const cx = info.offsetX + pt[0] * info.renderedW;
            const cy = info.offsetY + pt[1] * info.renderedH;

            // Halo pulsé
            ctx.beginPath();
            ctx.arc(cx, cy, 16, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(245, 158, 11, 0.4)";
            ctx.fill();

            // Point central
            ctx.beginPath();
            ctx.arc(cx, cy, 7, 0, Math.PI * 2);
            ctx.fillStyle = "#fbbf24";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Label
            ctx.fillStyle = "#fff";
            ctx.font = "bold 12px sans-serif";
            ctx.fillText(`Coin ${idx + 1}`, cx + 12, cy - 8);
        });

        // Lignes reliant les points cliqués
        if (clickedPointsNorm.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = "rgba(251, 191, 36, 0.85)";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            clickedPointsNorm.forEach((pt, i) => {
                const px = info.offsetX + pt[0] * info.renderedW;
                const py = info.offsetY + pt[1] * info.renderedH;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Réticule de visée sous la souris
        if (lastMousePos && lastMousePos.inside) {
            ctx.strokeStyle = "rgba(56, 189, 248, 0.9)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(lastMousePos.x, lastMousePos.y, 16, 0, Math.PI * 2);
            ctx.moveTo(lastMousePos.x - 24, lastMousePos.y);
            ctx.lineTo(lastMousePos.x + 24, lastMousePos.y);
            ctx.moveTo(lastMousePos.x, lastMousePos.y - 24);
            ctx.lineTo(lastMousePos.x, lastMousePos.y + 24);
            ctx.stroke();
        }
        return;
    }

    // 2. Rendu en mode Édition manuelle drag & drop
    if (editingCorners) {
        const labels = ["A8 (Haut-G)", "H8 (Haut-D)", "H1 (Bas-D)", "A1 (Bas-G)"];
        const colors = ["#f97316", "#38bdf8", "#22c55e", "#a855f7"];

        ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        currentCornersNorm.forEach((p, i) => {
            const x = info.offsetX + p[0] * info.renderedW;
            const y = info.offsetY + p[1] * info.renderedH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();

        currentCornersNorm.forEach((p, i) => {
            const x = info.offsetX + p[0] * info.renderedW;
            const y = info.offsetY + p[1] * info.renderedH;

            ctx.fillStyle = colors[i];
            ctx.beginPath();
            ctx.arc(x, y, 9, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px sans-serif";
            ctx.fillText(labels[i], x + 12, y + 4);
        });
    }
}

function toggleEditCorners() {
    editingCorners = !editingCorners;
    const btn = document.getElementById('btn-edit-corners');
    const canvas = document.getElementById('interactive-canvas');

    if (editingCorners) {
        guide4PointsMode = false;
        btn.classList.add('active');
        btn.innerText = '💾 Sauvegarder Coins';
        canvas.classList.add('interactive');
        resizeCanvas();
        drawOverlay();
    } else {
        btn.classList.remove('active');
        btn.innerText = '📐 Ajuster (Manuel)';
        canvas.classList.remove('interactive');
        postJSON('/api/vision/corners', { corners: currentCornersNorm });
        clearCanvas();
    }
}

function setupCanvasInteraction() {
    const canvas = document.getElementById('interactive-canvas');

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const info = getImageRenderInfo();

        if (info) {
            const inside = (mx >= info.offsetX && mx <= info.offsetX + info.renderedW &&
                            my >= info.offsetY && my <= info.offsetY + info.renderedH);
            lastMousePos = { x: mx, y: my, inside };
        }

        if (guide4PointsMode) {
            drawOverlay();
            return;
        }

        if (editingCorners && activeDragCorner !== null && info) {
            const imgX = Math.max(0, Math.min(mx - info.offsetX, info.renderedW));
            const imgY = Math.max(0, Math.min(my - info.offsetY, info.renderedH));

            currentCornersNorm[activeDragCorner] = [
                Math.round((imgX / info.renderedW) * 1000) / 1000,
                Math.round((imgY / info.renderedH) * 1000) / 1000
            ];
            drawOverlay();
        }
    });

    canvas.addEventListener('click', (e) => {
        if (!guide4PointsMode) return;
        const info = getImageRenderInfo();
        if (!info) return;

        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (mx < info.offsetX || mx > info.offsetX + info.renderedW ||
            my < info.offsetY || my > info.offsetY + info.renderedH) {
            return;
        }

        const normX = Math.round(((mx - info.offsetX) / info.renderedW) * 1000) / 1000;
        const normY = Math.round(((my - info.offsetY) / info.renderedH) * 1000) / 1000;

        clickedPointsNorm.push([normX, normY]);
        const count = clickedPointsNorm.length;
        const banner = document.getElementById('banner-action');

        if (count < 4) {
            banner.innerText = `🎯 Coin ${count} enregistré ! Cliquez sur le prochain coin (${count}/4)`;
            drawOverlay();
        } else {
            // Tri géométrique automatique (fonctionne quel que soit l'ordre des clics !)
            const ordered = sortCornersGeometrically(clickedPointsNorm);
            currentCornersNorm = ordered;
            guide4PointsMode = false;
            canvas.classList.remove('interactive');
            document.getElementById('btn-guide-4pts').classList.remove('active');
            banner.innerText = `✅ 4 coins enregistrés et ordonnés ! Cadrillage AR actualisé.`;
            drawOverlay();

            postJSON('/api/vision/corners', { corners: currentCornersNorm }).then(() => {
                setTimeout(clearCanvas, 1500);
            });
        }
    });

    canvas.addEventListener('mousedown', (e) => {
        if (!editingCorners || guide4PointsMode) return;
        const info = getImageRenderInfo();
        if (!info) return;

        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        activeDragCorner = null;
        currentCornersNorm.forEach((p, i) => {
            const px = info.offsetX + p[0] * info.renderedW;
            const py = info.offsetY + p[1] * info.renderedH;
            if (Math.hypot(mx - px, my - py) < 20) {
                activeDragCorner = i;
            }
        });
    });

    window.addEventListener('mouseup', () => {
        if (activeDragCorner !== null) {
            activeDragCorner = null;
            postJSON('/api/vision/corners', { corners: currentCornersNorm });
        }
    });
}

function postJSON(url, data) {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).then(r => r.json());
}
