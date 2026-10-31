// --- Variables globales de navigation dans l'historique (Chess.com style) ---
let liveGameStatus = null;
let browsingMoveIndex = null; // null = Direct (Live), 0 = Départ, 1..N = après coup N
let gameMoveHistory = [];
let initialGameFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
// État global frontend
let cameraConnected = false;
let show2DArrow = true;
let currentLastMoveUci = null;
let currentPendingBotMove = null;
let currentArrow = null;  // {uci, kind: 'bot' | 'hint'} fourni par le serveur
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
    checkCameraConnectionOnLoad();
    fetchVisionStatus();
    fetchGameStatus();
    initHistoryNavControls();
    initSavedGamesControls();
    setInterval(pollGameStatus, 1000);
    setInterval(pollLogs, 2000);
});

function checkCameraConnectionOnLoad() {
    fetch('/api/camera/status').then(r => r.json()).then(data => {
        if (data.settings) applyCameraSettingsToUI(data.settings);
        if (data.connected) {
            cameraConnected = true;
            const btn = document.getElementById('btn-toggle-cam');
            const imgStream = document.getElementById('cam-stream');
            const warpedStream = document.getElementById('warped-stream');
            const placeholder = document.getElementById('cam-placeholder');
            const warpedPlaceholder = document.getElementById('warped-placeholder');
            const badge = document.getElementById('cam-status-badge');

            if (btn) {
                btn.innerText = '⏹ Arrêter Caméra';
                btn.className = 'btn btn-secondary';
            }
            if (badge) {
                badge.innerText = 'Caméra En Direct';
                badge.className = 'badge badge-online';
            }
            if (imgStream && placeholder) {
                imgStream.src = `/api/camera/stream?t=${Date.now()}`;
                imgStream.style.display = 'block';
                placeholder.style.display = 'none';
            }
            if (warpedStream && warpedPlaceholder) {
                warpedStream.src = `/api/vision/warped_stream?t=${Date.now()}`;
                warpedStream.style.display = 'block';
                warpedPlaceholder.style.display = 'none';
            }
            setTimeout(resizeCanvas, 500);
        }
    });
}

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

function updateDigitalBoard(fen, lastMoveUci, checkInfo = null) {
    if (!fen) return;
    const parts = fen.split(' ')[0].split('/');
    let kingSquareMap = { 'K': null, 'k': null };

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
                if (ch === 'K') kingSquareMap['K'] = sqIdx;
                if (ch === 'k') kingSquareMap['k'] = sqIdx;
                col++;
            }
        }
    }

    // Effacer highlights et alertes échec
    document.querySelectorAll('.chess-sq').forEach(el => {
        el.classList.remove('highlight-from', 'highlight-to', 'king-check', 'king-checkmate');
    });

    currentLastMoveUci = lastMoveUci || null;
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

    // Coloration du Roi en Échec (Orange) ou Échec et Mat (Rouge)
    if (checkInfo) {
        let kingSq = checkInfo.checkSquare;
        if (kingSq === null || kingSq === undefined) {
            const turn = checkInfo.turn || (fen.split(' ')[1] === 'w' ? 'white' : 'black');
            kingSq = (turn === 'white') ? kingSquareMap['K'] : kingSquareMap['k'];
        }
        if (kingSq !== null && kingSq !== undefined) {
            const kingEl = document.getElementById(`sq-${kingSq}`);
            if (kingEl) {
                if (checkInfo.isCheckmate) {
                    kingEl.classList.add('king-checkmate');
                } else if (checkInfo.isCheck) {
                    kingEl.classList.add('king-check');
                }
            }
        }
    }

    render2DMoveArrow();
}

function render2DMoveArrow() {
    const arrowLine = document.getElementById('svg-move-arrow');
    if (!arrowLine) return;

    if (!show2DArrow) {
        arrowLine.style.display = 'none';
        return;
    }

    // Même flèche que les vues caméra : coup du bot à jouer ou conseil.
    // Disparaît dès que le coup est joué ou qu'une nouvelle partie démarre.
    const uci = currentArrow ? currentArrow.uci : null;
    if (!uci || uci.length < 4) {
        arrowLine.style.display = 'none';
        return;
    }

    const fCol = uci.charCodeAt(0) - 97;
    const fRow = parseInt(uci[1]) - 1;
    const tCol = uci.charCodeAt(2) - 97;
    const tRow = parseInt(uci[3]) - 1;

    // Cases 38x38 px : rangées inversées (7 - row)
    const sqSize = 38;
    const x1 = fCol * sqSize + sqSize / 2;
    const y1 = (7 - fRow) * sqSize + sqSize / 2;
    const x2 = tCol * sqSize + sqSize / 2;
    const y2 = (7 - tRow) * sqSize + sqSize / 2;

    const isBot = currentArrow.kind === 'bot';
    arrowLine.setAttribute('x1', x1);
    arrowLine.setAttribute('y1', y1);
    arrowLine.setAttribute('x2', x2);
    arrowLine.setAttribute('y2', y2);
    arrowLine.setAttribute('stroke', isBot ? '#eab308' : '#38bdf8');
    arrowLine.setAttribute('marker-end', isBot ? 'url(#arrowhead-bot)' : 'url(#arrowhead)');
    arrowLine.style.display = 'block';
}

function initEventListeners() {
    // Caméra
    document.getElementById('btn-detect-cam').addEventListener('click', detectCameras);
    document.getElementById('btn-toggle-cam').addEventListener('click', toggleCamera);
    document.getElementById('input-upload-photo').addEventListener('change', uploadPhoto);
    document.getElementById('btn-rotate-cam').addEventListener('click', function() {
        postJSON('/api/camera/settings', { toggle_rotate_180: true }).then(res => {
            if (res.status && res.status.settings) {
                this.classList.toggle('active', res.status.settings.rotate_180);
            }
        });
    });
    document.getElementById('btn-flip-cam').addEventListener('click', function() {
        postJSON('/api/camera/settings', { toggle_flip_h: true }).then(res => {
            if (res.status && res.status.settings) {
                this.classList.toggle('active', res.status.settings.flip_h);
            }
        });
    });
    document.getElementById('btn-snapshot').addEventListener('click', () => {
        postJSON('/api/camera/snapshot', {});
    });

    // Filtres Vision Multiples & Cumulables (réductible / déroulable)
    const btnToggleFilters = document.getElementById('btn-toggle-filters');
    const filtersToolbar = document.getElementById('filters-toolbar');
    if (btnToggleFilters && filtersToolbar) {
        btnToggleFilters.addEventListener('click', function() {
            this.classList.toggle('active');
            const isOpen = this.classList.contains('active');
            filtersToolbar.style.display = isOpen ? 'flex' : 'none';
            const count = currentFilters.length > 0 ? ` (${currentFilters.length})` : '';
            this.innerText = isOpen ? `🎨 Masquer Filtres ▴` : `🎨 Filtres${count} ▾`;
        });
    }

    let currentFilters = [];
    document.querySelectorAll('.btn-filter').forEach(btn => {
        btn.addEventListener('click', function() {
            const fName = this.dataset.filter;
            this.classList.toggle('active');
            if (this.classList.contains('active')) {
                if (!currentFilters.includes(fName)) currentFilters.push(fName);
            } else {
                currentFilters = currentFilters.filter(f => f !== fName);
            }
            if (btnToggleFilters && !btnToggleFilters.classList.contains('active')) {
                const count = currentFilters.length > 0 ? ` (${currentFilters.length})` : '';
                btnToggleFilters.innerText = `🎨 Filtres${count} ▾`;
            }
            postJSON('/api/camera/settings', { active_filters: currentFilters });
        });
    });

    document.getElementById('btn-filter-reset').addEventListener('click', function() {
        currentFilters = [];
        document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
        if (btnToggleFilters) {
            const isOpen = btnToggleFilters.classList.contains('active');
            btnToggleFilters.innerText = isOpen ? `🎨 Masquer Filtres ▴` : `🎨 Filtres ▾`;
        }
        postJSON('/api/camera/settings', { active_filters: [] });
    });

    // Vision & Calibration
    document.getElementById('btn-auto-calib').addEventListener('click', autoCalibrate);
    const chkContinuous = document.getElementById('chk-continuous-calib');
    if (chkContinuous) {
        chkContinuous.addEventListener('change', (e) => {
            const enabled = e.target.checked;
            postJSON('/api/vision/continuous-detection', { enabled })
                .then(res => {
                    const banner = document.getElementById('banner-action');
                    if (banner) {
                        banner.innerText = enabled 
                            ? '🔄 Détection continue activée (réajustement auto toutes les 5s)'
                            : '⏸️ Détection continue désactivée';
                    }
                })
                .catch(err => {
                    console.error('Erreur toggle détection continue:', err);
                });
        });
    }
    document.getElementById('btn-guide-4pts').addEventListener('click', start4PointsGuide);
    document.getElementById('btn-rotate-board').addEventListener('click', () => {
        postJSON('/api/vision/rotate-board', {}).then(res => {
            if (res.corners) {
                currentCornersNorm = res.corners;
                drawOverlay();
            }
        });
    });
    document.getElementById('btn-edit-corners').addEventListener('click', toggleEditCorners);

    document.getElementById('btn-toggle-aruco').addEventListener('click', function() {
        this.classList.toggle('active');
        postJSON('/api/vision/overlay-settings', { show_aruco_debug: this.classList.contains('active') });
    });
    document.getElementById('btn-toggle-grid').addEventListener('click', function() {
        this.classList.toggle('active');
        postJSON('/api/vision/overlay-settings', { show_ar_grid: this.classList.contains('active') });
    });
    document.getElementById('btn-toggle-labels').addEventListener('click', function() {
        this.classList.toggle('active');
        postJSON('/api/vision/overlay-settings', { show_ar_labels: this.classList.contains('active') });
    });
    const btnToggleArArrow = document.getElementById('btn-toggle-ar-arrow');
    if (btnToggleArArrow) {
        btnToggleArArrow.addEventListener('click', function() {
            this.classList.toggle('active');
            postJSON('/api/vision/overlay-settings', { show_ar_arrow: this.classList.contains('active') });
        });
    }

    const btnToggleVisionDebug = document.getElementById('btn-toggle-vision-debug');
    if (btnToggleVisionDebug) {
        btnToggleVisionDebug.addEventListener('click', function() {
            this.classList.toggle('active');
            postJSON('/api/vision/overlay-settings', { show_ar_vision_debug: this.classList.contains('active') });
        });
    }

    const btnTogglePieces = document.getElementById('btn-toggle-pieces');
    if (btnTogglePieces) {
        btnTogglePieces.addEventListener('click', function() {
            this.classList.toggle('active');
            postJSON('/api/vision/overlay-settings', { show_ar_pieces: this.classList.contains('active') });
        });
    }

    const btnToggleWarpedGrid = document.getElementById('btn-toggle-warped-grid');
    if (btnToggleWarpedGrid) {
        btnToggleWarpedGrid.addEventListener('click', function() {
            this.classList.toggle('active');
            postJSON('/api/vision/overlay-settings', { show_warped_grid: this.classList.contains('active') });
        });
    }

    const btnToggleWarpedArrow = document.getElementById('btn-toggle-warped-arrow');
    if (btnToggleWarpedArrow) {
        btnToggleWarpedArrow.addEventListener('click', function() {
            this.classList.toggle('active');
            postJSON('/api/vision/overlay-settings', { show_warped_arrow: this.classList.contains('active') });
        });
    }

    const btnToggle2DArrow = document.getElementById('btn-toggle-2d-arrow');
    if (btnToggle2DArrow) {
        btnToggle2DArrow.addEventListener('click', function() {
            this.classList.toggle('active');
            show2DArrow = this.classList.contains('active');
            render2DMoveArrow();
        });
    }

    const settingButtonMap = {
        'btn-toggle-aruco': 'show_aruco_debug',
        'btn-toggle-grid': 'show_ar_grid',
        'btn-toggle-labels': 'show_ar_labels',
        'btn-toggle-ar-arrow': 'show_ar_arrow',
        'btn-toggle-vision-debug': 'show_ar_vision_debug',
        'btn-toggle-pieces': 'show_ar_pieces',
        'btn-toggle-warped-grid': 'show_warped_grid',
        'btn-toggle-warped-arrow': 'show_warped_arrow'
    };
    window.applyVisionSettingsToUI = function(settings) {
        if (!settings) return;
        for (const [btnId, settingKey] of Object.entries(settingButtonMap)) {
            const btn = document.getElementById(btnId);
            if (btn && settings[settingKey] !== undefined) {
                btn.classList.toggle('active', !!settings[settingKey]);
            }
        }
    };

    const btnToggleConsole = document.getElementById('btn-toggle-console');
    if (btnToggleConsole) {
        btnToggleConsole.addEventListener('click', function() {
            this.classList.toggle('active');
            const panel = document.getElementById('detection-console-panel');
            if (panel) {
                const isOpen = this.classList.contains('active');
                panel.style.display = isOpen ? 'block' : 'none';
                this.innerText = isOpen ? '🎛️ Masquer Config ▴' : '🎛️ Console Config ▾';
            }
        });
    }

    setupDetectionConsole();

    // Contrôles Partie
    document.getElementById('btn-new-game').addEventListener('click', () => {
        postJSON('/api/game/new', {}).then(pollGameStatus);
    });
    document.getElementById('btn-confirm-setup').addEventListener('click', () => {
        postJSON('/api/game/confirm-setup', {}).then(pollGameStatus);
    });
    document.getElementById('btn-resync').addEventListener('click', () => {
        postJSON('/api/game/resync-reference', {}).then(res => {
            if (res && !res.success && res.message) {
                document.getElementById('banner-action').innerText = `⚠️ ${res.message}`;
            } else {
                pollGameStatus();
            }
        });
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

function applyCameraSettingsToUI(settings) {
    if (!settings) return;
    const btnRot = document.getElementById('btn-rotate-cam');
    if (btnRot) btnRot.classList.toggle('active', !!settings.rotate_180);
    const btnFlip = document.getElementById('btn-flip-cam');
    if (btnFlip) btnFlip.classList.toggle('active', !!settings.flip_h);
}

function detectCameras() {
    fetch('/api/camera/list')
        .then(r => r.json())
        .then(data => {
            if (data.status && data.status.settings) {
                applyCameraSettingsToUI(data.status.settings);
            }
            const sel = document.getElementById('camera-select');
            sel.innerHTML = '';
            if (data.cameras && data.cameras.length > 0) {
                const targetIdx = (data.status && data.status.index !== undefined && data.status.index !== null) ? data.status.index : 0;
                data.cameras.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c.index;
                    opt.text = c.label;
                    if (c.index === targetIdx) opt.selected = true;
                    else if (c.active && (targetIdx === null || targetIdx === undefined)) opt.selected = true;
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

function updateArucoBadge(idsList) {
    const badge = document.getElementById('aruco-count-badge');
    if (!badge) return;
    const count = Array.isArray(idsList) ? idsList.length : 0;
    badge.innerText = `Marqueurs: ${count}/4 ${count > 0 ? '[' + idsList.map(i => '#' + i).join(',') + ']' : ''}`;
    badge.className = count >= 4 ? 'badge badge-online' : 'badge badge-offline';
}

let detConfigTimer = null;
function setupDetectionConsole() {
    const sliders = [
        { id: 'cfg-inset-pct', valId: 'val-inset-pct', key: 'aruco_inset_pct', unit: ' %', isFloat: true },
        { id: 'cfg-min-side',  valId: 'val-min-side',  key: 'min_side_px',     unit: ' px', isFloat: false },
        { id: 'cfg-max-side',  valId: 'val-max-side',  key: 'max_side_px',     unit: ' px', isFloat: false },
        { id: 'cfg-clahe-clip',valId: 'val-clahe-clip',key: 'clahe_clip',      unit: '',    isFloat: true },
        { id: 'cfg-bin-thresh',valId: 'val-bin-thresh',key: 'bin_thresh',      unit: '',    isFloat: false },
        { id: 'cfg-border-err',valId: 'val-border-err',key: 'border_err',      unit: '',    isFloat: true },
        { id: 'cfg-pastille-min-area', valId: 'val-pastille-min-area', key: 'pastille_min_area', unit: ' px', isFloat: false },
        { id: 'cfg-pastille-max-area', valId: 'val-pastille-max-area', key: 'pastille_max_area', unit: ' px', isFloat: false },
        { id: 'cfg-pastille-min-circ', valId: 'val-pastille-min-circ', key: 'pastille_min_circ', unit: '',    isFloat: true },
        { id: 'cfg-pastille-min-sat',  valId: 'val-pastille-min-sat',  key: 'pastille_min_sat',  unit: '',    isFloat: false },
        { id: 'cfg-parallax-scale',    valId: 'val-parallax-scale',    key: 'parallax_scale',    unit: '',    isFloat: true }
    ];

    function pushConfigUpdate() {
        const payload = {};
        sliders.forEach(s => {
            const el = document.getElementById(s.id);
            if (el) {
                payload[s.key] = s.isFloat ? parseFloat(el.value) : parseInt(el.value);
            }
        });

        postJSON('/api/vision/detection-config', payload).then(res => {
            if (res && res.success) {
                if (res.corners) {
                    currentCornersNorm = res.corners;
                    drawOverlay();
                }
                if (res.detected_aruco_ids) {
                    updateArucoBadge(res.detected_aruco_ids);
                    const c = res.detected_aruco_ids.length;
                    document.getElementById('banner-action').innerText =
                        `🎛️ Paramètres appliqués en direct — ${c}/4 marqueurs détectés (Inset: ${payload.aruco_inset_pct}%).`;
                }
            }
        });
    }

    sliders.forEach(s => {
        const el = document.getElementById(s.id);
        const valEl = document.getElementById(s.valId);
        if (!el || !valEl) return;
        el.addEventListener('input', () => {
            valEl.innerText = `${el.value}${s.unit}`;
            if (detConfigTimer) clearTimeout(detConfigTimer);
            detConfigTimer = setTimeout(pushConfigUpdate, 120);
        });
    });

    // Sélecteur de mode de détection des pièces
    const selPieceMode = document.getElementById('cfg-piece-mode');
    if (selPieceMode) {
        selPieceMode.addEventListener('change', () => {
            postJSON('/api/vision/detection-config', { piece_mode: selPieceMode.value });
        });
    }

    // Sélecteur de méthode de détection des coins (barre d'outils + console synchronisés)
    const selMethodToolbar = document.getElementById('select-det-method');
    const selMethodConsole = document.getElementById('cfg-det-method');

    window.applyDetectionConfigToUI = function(cfg) {
        if (!cfg) return;
        sliders.forEach(s => {
            const el = document.getElementById(s.id);
            const valEl = document.getElementById(s.valId);
            if (el && cfg[s.key] !== undefined) {
                if (document.activeElement !== el) {
                    el.value = cfg[s.key];
                    if (valEl) valEl.innerText = `${cfg[s.key]}${s.unit}`;
                }
            }
        });
        if (cfg.detection_method) {
            if (selMethodToolbar && document.activeElement !== selMethodToolbar) selMethodToolbar.value = cfg.detection_method;
            if (selMethodConsole && document.activeElement !== selMethodConsole) selMethodConsole.value = cfg.detection_method;
        }
        if (cfg.piece_mode && selPieceMode && document.activeElement !== selPieceMode) {
            selPieceMode.value = cfg.piece_mode;
        }
    };

    function onMethodChange(newVal) {
        if (selMethodToolbar) selMethodToolbar.value = newVal;
        if (selMethodConsole) selMethodConsole.value = newVal;
        postJSON('/api/vision/detection-config', { detection_method: newVal }).then(res => {
            if (res && res.success) {
                if (res.corners) {
                    currentCornersNorm = res.corners;
                    drawOverlay();
                }
                const labelMap = { 
                    'COUPLED': 'ArUco + Hough (Contour ArUco + Lignes Hough)', 
                    'ARUCO': 'ArUco seul (Grille uniforme 8x8)', 
                    'HOUGH': 'Hough seul (Sans marqueurs)' 
                };
                const nLines = res.status && res.status.grid_refined_lines;
                const linesDetail = (newVal === 'COUPLED' && nLines) ? ` (${nLines}/14 lignes ajustées)` : '';
                document.getElementById('banner-action').innerText =
                    `🎯 Méthode activée : ${labelMap[newVal] || newVal}${linesDetail}. Enregistrée.`;
            }
        });
    }
    if (selMethodToolbar) selMethodToolbar.addEventListener('change', (e) => onMethodChange(e.target.value));
    if (selMethodConsole) selMethodConsole.addEventListener('change', (e) => onMethodChange(e.target.value));

    const btnReset = document.getElementById('btn-reset-det-config');
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            postJSON('/api/vision/detection-config/reset', {}).then(res => {
                if (res && res.success) {
                    if (res.config && window.applyDetectionConfigToUI) {
                        window.applyDetectionConfigToUI(res.config);
                    }
                    if (res.corners) {
                        currentCornersNorm = res.corners;
                        drawOverlay();
                    }
                    if (res.detected_aruco_ids) {
                        updateArucoBadge(res.detected_aruco_ids);
                    }
                    document.getElementById('banner-action').innerText =
                        '↺ Paramètres réinitialisés aux valeurs par défaut d\'usine et enregistrés.';
                }
            });
        });
    }

    const btnCalibPieces = document.getElementById('btn-auto-calib-pieces');
    if (btnCalibPieces) {
        btnCalibPieces.addEventListener('click', () => {
            btnCalibPieces.disabled = true;
            btnCalibPieces.innerText = '⏳ Analyse départ...';
            document.getElementById('banner-action').innerText =
                '🎯 Analyse des 4 rangées de départ (Vert, Rouge, Violet, Bleu) et des cases vides...';
            postJSON('/api/vision/auto-calibrate-pieces', {}).then(res => {
                btnCalibPieces.disabled = false;
                btnCalibPieces.innerText = '🎯 Calibrer Couleurs Départ';
                if (res && res.success) {
                    document.getElementById('banner-action').innerText = `✅ ${res.message}`;
                    if (res.summary && res.summary.total_pastilles_found !== undefined) {
                        const pBadge = document.getElementById('pastille-count-badge');
                        if (pBadge) pBadge.innerText = `Pastilles: ${res.summary.total_pastilles_found}/32`;
                    }
                } else {
                    document.getElementById('banner-action').innerText = `⚠️ ${res.error || 'Erreur calibration pièces'}`;
                }
            });
        });
    }
}

function autoCalibrate() {
    const btn = document.getElementById('btn-auto-calib');
    const banner = document.getElementById('banner-action');
    const originalText = '🎯 Détection';
    
    btn.disabled = true;
    btn.innerText = '⏳ Analyse ArUco & Grille...';
    banner.innerText = '⏳ Recherche des marqueurs ArUco & petits carrés N/B aux 4 coins...';

    postJSON('/api/vision/auto-calibrate', {})
        .then(res => {
            btn.disabled = false;
            btn.innerText = originalText;
            if (res.detected_aruco_ids) {
                updateArucoBadge(res.detected_aruco_ids);
            }
            if (res.success && res.corners) {
                currentCornersNorm = res.corners;
                drawOverlay();
                banner.innerText = `✅ Succès : ${res.message}`;
            } else {
                banner.innerText = `⚠️ ${res.message || "Aucun marqueur détecté"}. Ajustez la Console Config ci-dessous ou cliquez sur « 🎯 Pointer 4 Clics ».`;
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
            if (data.corners_norm && !editingCorners && !guide4PointsMode) {
                currentCornersNorm = data.corners_norm;
                drawOverlay();
            }
            if (data.detected_aruco_ids) {
                updateArucoBadge(data.detected_aruco_ids);
            }
            if (data.pastilles_detected_count !== undefined) {
                const pBadge = document.getElementById('pastille-count-badge');
                if (pBadge) {
                    pBadge.innerText = `Pastilles: ${data.pastilles_detected_count}/32`;
                    pBadge.className = data.pastilles_detected_count >= 16 ? 'badge badge-online' : 'badge';
                }
            }
            if (data.continuous_detection !== undefined) {
                const chk = document.getElementById('chk-continuous-calib');
                if (chk && document.activeElement !== chk) {
                    chk.checked = !!data.continuous_detection;
                }
            }
            if (data.detection_config && window.applyDetectionConfigToUI) {
                window.applyDetectionConfigToUI(data.detection_config);
            }
            if (data.settings && window.applyVisionSettingsToUI) {
                window.applyVisionSettingsToUI(data.settings);
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
            if (data.settings) applyCameraSettingsToUI(data.settings);
        });
        fetchVisionStatus();
    }
    fetchGameStatus();
}

function updateCheckAlertBanner(g) {
    const banner = document.getElementById('check-alert-banner');
    const icon = document.getElementById('check-banner-icon');
    const title = document.getElementById('check-banner-title');
    const subtitle = document.getElementById('check-banner-subtitle');
    if (!banner || !icon || !title || !subtitle) return;

    if (g.is_checkmate) {
        banner.style.display = 'flex';
        banner.className = 'check-banner banner-mat is-active';
        icon.innerText = '👑';
        const winnerName = g.winner === 'white' ? 'BLANCS' : (g.winner === 'black' ? 'NOIRS' : 'NULLE');
        const loserName = g.winner === 'white' ? 'Noirs' : 'Blancs';
        title.innerText = `ÉCHEC ET MAT ! LES ${winnerName} GAGNENT ! 🏆`;
        subtitle.innerText = `Le Roi des ${loserName} est mat. Fin de partie.`;
    } else if (g.is_check) {
        banner.style.display = 'flex';
        banner.className = 'check-banner banner-check is-active';
        icon.innerText = '⚠️';
        const kingColor = g.turn === 'white' ? 'Blancs' : 'Noirs';
        title.innerText = `ÉCHEC AU ROI (${kingColor.toUpperCase()}) !`;
        subtitle.innerText = `Le Roi des ${kingColor} est attaqué ! Vous devez parer l'échec.`;
    } else {
        banner.style.display = 'flex';
        banner.className = 'check-banner';
    }
}

let historyNavInitialized = false;

function updateHistoryNavButtons() {
    const btnStart = document.getElementById('btn-hist-start');
    const btnPrev = document.getElementById('btn-hist-prev');
    const btnNext = document.getElementById('btn-hist-next');
    const btnEnd = document.getElementById('btn-hist-end');
    const statusLabel = document.getElementById('hist-status-label');
    const liveDot = document.querySelector('.live-dot');

    const totalMoves = gameMoveHistory.length;
    const isLive = (browsingMoveIndex === null);

    if (btnStart) btnStart.disabled = (totalMoves === 0 || browsingMoveIndex === 0);
    if (btnPrev) btnPrev.disabled = (totalMoves === 0 || browsingMoveIndex === 0);
    if (btnNext) btnNext.disabled = (totalMoves === 0 || isLive);
    if (btnEnd) {
        btnEnd.disabled = (totalMoves === 0 || isLive);
        if (isLive) {
            btnEnd.classList.add('is-live');
        } else {
            btnEnd.classList.remove('is-live');
        }
    }

    if (statusLabel) {
        if (isLive) {
            statusLabel.innerText = "En Direct (Live)";
            statusLabel.style.color = "#38bdf8";
        } else if (browsingMoveIndex === 0) {
            statusLabel.innerText = "Position initiale (Départ)";
            statusLabel.style.color = "#f59e0b";
        } else {
            const m = gameMoveHistory[browsingMoveIndex - 1];
            statusLabel.innerText = `Coup ${browsingMoveIndex}/${totalMoves} (${m ? m.san : ''})`;
            statusLabel.style.color = "#f59e0b";
        }
    }

    if (liveDot) {
        if (isLive) {
            liveDot.classList.remove('hist-mode');
        } else {
            liveDot.classList.add('hist-mode');
        }
    }

    // Mettre à jour la surbrillance dans l'historique PGN
    document.querySelectorAll('.move-item').forEach(el => {
        const idx = parseInt(el.dataset.moveIdx);
        if (idx === browsingMoveIndex) {
            el.classList.add('active-move');
        } else {
            el.classList.remove('active-move');
        }
    });
}

function displayBrowsedMove(index) {
    const total = gameMoveHistory.length;
    if (total === 0 && index !== 0) return;
    browsingMoveIndex = Math.max(0, Math.min(index, total));

    if (browsingMoveIndex === 0) {
        // Position initiale : jamais d'échec ni mat
        updateDigitalBoard(initialGameFen, null, null);
        updateCheckAlertBanner({ is_check: false, is_checkmate: false });
    } else {
        const moveEntry = gameMoveHistory[browsingMoveIndex - 1];
        if (moveEntry) {
            const isMat = !!(moveEntry.is_checkmate || (moveEntry.san && moveEntry.san.includes('#')));
            const isChk = isMat || !!(moveEntry.is_check || (moveEntry.san && moveEntry.san.includes('+')));
            const turn = moveEntry.turn || (moveEntry.fen ? (moveEntry.fen.split(' ')[1] === 'w' ? 'white' : 'black') : 'white');
            let winner = moveEntry.winner || null;
            if (isMat && !winner) {
                // Le joueur qui a joué le coup fatal gagne
                winner = (turn === 'white') ? 'black' : 'white';
            }

            const checkInfo = {
                is_check: isChk,
                is_checkmate: isMat,
                turn: turn,
                winner: winner,
                check_square: moveEntry.check_square || null
            };

            // Échiquier 2D avec coloration orange (échec) ou rouge (mat) seulement si le roi l'est à ce coup
            updateDigitalBoard(moveEntry.fen, moveEntry.uci, {
                isCheck: isChk,
                isCheckmate: isMat,
                checkSquare: moveEntry.check_square || null,
                turn: turn
            });

            // Message d'alerte synchronisé avec le coup du passé
            updateCheckAlertBanner(checkInfo);
        }
    }

    updateHistoryNavButtons();
}

function goLive() {
    browsingMoveIndex = null;
    if (liveGameStatus) {
        updateGameUI({ game: liveGameStatus });
    }
    updateHistoryNavButtons();
}

function initHistoryNavControls() {
    if (historyNavInitialized) return;
    historyNavInitialized = true;

    const btnStart = document.getElementById('btn-hist-start');
    const btnPrev = document.getElementById('btn-hist-prev');
    const btnNext = document.getElementById('btn-hist-next');
    const btnEnd = document.getElementById('btn-hist-end');

    if (btnStart) {
        btnStart.addEventListener('click', (e) => {
            e.preventDefault();
            displayBrowsedMove(0);
        });
    }

    if (btnPrev) {
        btnPrev.addEventListener('click', (e) => {
            e.preventDefault();
            const total = gameMoveHistory.length;
            if (total === 0) return;
            const current = (browsingMoveIndex === null) ? total : browsingMoveIndex;
            if (current > 0) {
                displayBrowsedMove(current - 1);
            }
        });
    }

    if (btnNext) {
        btnNext.addEventListener('click', (e) => {
            e.preventDefault();
            if (browsingMoveIndex === null) return;
            const total = gameMoveHistory.length;
            if (browsingMoveIndex >= total) {
                goLive();
            } else if (browsingMoveIndex === total - 1) {
                displayBrowsedMove(total);
            } else {
                displayBrowsedMove(browsingMoveIndex + 1);
            }
        });
    }

    if (btnEnd) {
        btnEnd.addEventListener('click', (e) => {
            e.preventDefault();
            goLive();
        });
    }

    // Navigation clavier ergonomique : Flèches Gauche/Droite
    document.addEventListener('keydown', (e) => {
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const total = gameMoveHistory.length;
            if (total === 0) return;
            const current = (browsingMoveIndex === null) ? total : browsingMoveIndex;
            if (current > 0) {
                displayBrowsedMove(current - 1);
            }
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            if (browsingMoveIndex === null) return;
            const total = gameMoveHistory.length;
            if (browsingMoveIndex >= total) {
                goLive();
            } else if (browsingMoveIndex === total - 1) {
                displayBrowsedMove(total);
            } else {
                displayBrowsedMove(browsingMoveIndex + 1);
            }
        } else if (e.key === 'Home') {
            e.preventDefault();
            displayBrowsedMove(0);
        } else if (e.key === 'End') {
            e.preventDefault();
            goLive();
        }
    });
}

function updateGameUI(data) {
    if (!data || !data.game) return;
    if (typeof isPositionEditing === 'function' && isPositionEditing()) return;
    const g = data.game;
    liveGameStatus = g;
    gameMoveHistory = g.history || [];
    if (g.initial_fen) initialGameFen = g.initial_fen;

    currentPendingBotMove = g.pending_bot_move;
    currentArrow = g.arrow || null;

    // Alerte bannière Échec & Échiquier : mis à jour SEULEMENT si on est en direct (Live)
    // pour ne pas écraser l'affichage du coup que l'utilisateur est en train d'examiner
    if (browsingMoveIndex === null) {
        updateCheckAlertBanner(g);
        updateDigitalBoard(g.fen, g.last_move, {
            isCheck: g.is_check,
            isCheckmate: g.is_checkmate,
            checkSquare: g.check_square,
            turn: g.turn
        });
    }

    const turnBadge = document.getElementById('turn-badge');
    const banner = document.getElementById('banner-action');

    turnBadge.innerText = g.state;
    if (g.is_checkmate) {
        turnBadge.className = 'badge badge-offline';
        const win = g.winner === 'white' ? 'Blancs' : 'Noirs';
        banner.innerText = `👑 Échec et Mat ! Victoire des ${win} ! Partie terminée.`;
    } else if (g.is_check) {
        turnBadge.className = 'badge badge-online';
        banner.innerText = `⚠️ Échec au Roi (${g.turn === 'white' ? 'Blancs' : 'Noirs'}) ! Déplacez votre Roi ou parez l'échec.`;
    } else if (g.ambiguity) {
        banner.innerText = `❓ La caméra hésite entre : ${g.ambiguity}. Placez la bonne position via « ✏️ Corriger la position ».`;
    } else if (g.vision_hint) {
        banner.innerText = `⏸️ Coup non validé : ${g.vision_hint}.`;
    } else if (g.state === 'WAIT_HUMAN_MOVE') {
        turnBadge.className = 'badge badge-online';
        banner.innerText = `🟢 À votre tour de jouer (${g.turn === 'white' ? 'Blancs' : 'Noirs'}). Déplacez une pièce sur le plateau.`;
    } else if (g.state === 'WAIT_BOT_EXECUTION' && g.pending_bot_move) {
        turnBadge.className = 'badge badge-offline';
        banner.innerText = `🤖 Déplacez la pièce du Bot : ${g.pending_bot_move.san} (${g.pending_bot_move.uci}) sur votre échiquier !`;
    } else if (g.state === 'SETUP_CHECK') {
        banner.innerText = `♟️ Placez les 32 pièces sur les rangées indiquées en AR, puis cliquez sur « ✅ Valider Pièces Initiales ».`;
    }

    const btnConfirmSetup = document.getElementById('btn-confirm-setup');
    if (btnConfirmSetup) {
        const canConfirm = (g.state === 'SETUP_CHECK' || g.state === 'IDLE') && (!g.history || g.history.length === 0);
        btnConfirmSetup.disabled = !canConfirm;
        btnConfirmSetup.title = canConfirm 
            ? "Valider la position initiale des 32 pièces pour lancer la partie"
            : "Partie en cours : utilisez « 📸 Resync (J'adoube) » ou « ✏️ Corriger la position »";
    }

    // Historique PGN cliquable
    const historyBox = document.getElementById('move-history-box');
    if (historyBox) {
        if (gameMoveHistory.length > 0) {
            historyBox.innerHTML = gameMoveHistory.map((m, idx) => {
                const moveNum = (idx + 1);
                const isActive = (browsingMoveIndex === moveNum) ? 'active-move' : '';
                return `<span class="move-item ${isActive}" data-move-idx="${moveNum}" onclick="displayBrowsedMove(${moveNum})">#${m.num} <b>${m.san}</b> (${m.player})</span>`;
            }).join(' • ');
        } else {
            historyBox.innerHTML = '<em>Historique des coups PGN vide</em>';
        }
    }

    updateHistoryNavButtons();
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


// --- Gestionnaire d'Archives et de Parties Sauvegardées ---
function initSavedGamesControls() {
    const btnOpen = document.getElementById('btn-open-saved-games');
    const btnClose = document.getElementById('btn-close-saved-modal');
    const modal = document.getElementById('modal-saved-games');
    const btnRefresh = document.getElementById('btn-refresh-saved-list');
    const btnSaveCheckpoint = document.getElementById('btn-save-current-checkpoint');

    if (btnOpen) {
        btnOpen.addEventListener('click', () => {
            openSavedGamesModal();
        });
    }

    if (btnClose && modal) {
        btnClose.addEventListener('click', () => {
            modal.style.display = 'none';
        });
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = 'none';
        });
    }

    if (btnRefresh) {
        btnRefresh.addEventListener('click', () => {
            fetchSavedGamesList();
        });
    }

    if (btnSaveCheckpoint) {
        btnSaveCheckpoint.addEventListener('click', () => {
            saveCurrentGameCheckpoint();
        });
    }
}

function openSavedGamesModal() {
    const modal = document.getElementById('modal-saved-games');
    if (!modal) return;
    modal.style.display = 'flex';
    fetchSavedGamesList();
}

function fetchSavedGamesList() {
    const container = document.getElementById('saved-games-list');
    if (!container) return;
    container.innerHTML = '<div style="text-align: center; color: #94a3b8; padding: 20px;">Chargement des parties sauvegardées...</div>';

    fetch('/api/game/saved')
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.games || data.games.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #94a3b8; padding: 30px;">Aucune partie archivée pour le moment.<br><small>Toute partie jouée est automatiquement sauvegardée ici !</small></div>';
                return;
            }

            container.innerHTML = data.games.map(g => {
                const movesStr = (g.preview_moves && g.preview_moves.length > 0)
                    ? `Derniers coups : ${g.preview_moves.join(' ')}`
                    : 'Aucun coup joué';
                const isCurrent = (liveGameStatus && liveGameStatus.game_id === g.game_id);
                const currentBadge = isCurrent ? '<span class="badge badge-online" style="font-size: 0.7rem; margin-left: 6px;">Active</span>' : '';

                return `
                <div class="saved-game-card">
                    <div class="saved-game-info">
                        <div class="saved-game-title">
                            📅 ${g.date_str} ${currentBadge}
                        </div>
                        <div class="saved-game-meta">
                            ♟️ <b>${g.moves_count} coups</b> • État : ${g.state || 'En cours'}
                        </div>
                        <div class="saved-game-preview" title="${movesStr}">
                            ${movesStr}
                        </div>
                    </div>
                    <div class="saved-game-actions">
                        <button class="btn btn-sm btn-primary" onclick="loadSavedGameById('${g.game_id}')" title="Restaurer cette position sur le plateau">🔄 Charger</button>
                        <a href="/api/game/saved/export-pgn?game_id=${g.game_id}" class="btn btn-sm" download="${g.game_id}.pgn" title="Télécharger le fichier PGN">⬇ PGN</a>
                        <button class="btn btn-sm" style="color: #ef4444; border-color: rgba(239, 68, 68, 0.4);" onclick="deleteSavedGameById('${g.game_id}')" title="Supprimer de l'archive">🗑️</button>
                    </div>
                </div>
                `;
            }).join('');
        })
        .catch(err => {
            container.innerHTML = `<div style="text-align: center; color: #ef4444; padding: 20px;">Erreur lors du chargement : ${err.message}</div>`;
        });
}

function loadSavedGameById(gameId) {
    if (!confirm(`Voulez-vous restaurer la partie archivée (${gameId}) ?\nL'échiquier numérique et l'IA se caleront immédiatement dessus.`)) {
        return;
    }

    fetch('/api/game/saved/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            alert('✅ Partie restaurée avec succès !');
            const modal = document.getElementById('modal-saved-games');
            if (modal) modal.style.display = 'none';
            goLive();
            fetchGameStatus();
        } else {
            alert(`❌ Erreur : ${data.message || 'Impossible de restaurer la partie'}`);
        }
    })
    .catch(err => {
        alert(`❌ Erreur de communication : ${err.message}`);
    });
}

function deleteSavedGameById(gameId) {
    if (!confirm(`Supprimer définitivement la sauvegarde ${gameId} ?`)) {
        return;
    }

    fetch('/api/game/saved/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            fetchSavedGamesList();
        } else {
            alert(`❌ Erreur : ${data.message}`);
        }
    });
}

function saveCurrentGameCheckpoint() {
    fetch('/api/game/saved/save-checkpoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            alert('💾 Partie en cours archivée avec succès !');
            fetchSavedGamesList();
        } else {
            alert('❌ Erreur lors de la sauvegarde');
        }
    });
}
