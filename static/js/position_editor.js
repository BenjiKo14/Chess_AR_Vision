// Éditeur de position : permet de corriger à la main la position connue
// quand elle ne correspond plus au vrai échiquier.
// Dépend de app.js (PIECE_UNICODE, postJSON, pollGameStatus).

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const positionEditor = {
    active: false,
    pieces: new Array(64).fill(null),  // index = case python-chess (a1 = 0, h8 = 63)
    tool: 'move',                       // 'move', 'x' (effacer) ou une lettre de pièce
    pickedSq: null,                     // case sélectionnée en mode déplacement
    fullmove: 1,
};

// Appelé par app.js : tant que l'éditeur est ouvert, le rafraîchissement
// périodique ne doit pas écraser la position en cours de correction.
function isPositionEditing() {
    return positionEditor.active;
}

function editorLoadFen(fen) {
    const [placement, turn, , , , fullmove] = fen.split(' ');
    positionEditor.pieces.fill(null);
    placement.split('/').forEach((row, r) => {
        let col = 0;
        for (const ch of row) {
            if (ch >= '1' && ch <= '8') {
                col += parseInt(ch);
            } else {
                positionEditor.pieces[(7 - r) * 8 + col] = ch;
                col++;
            }
        }
    });
    document.getElementById('editor-turn').value = turn === 'b' ? 'b' : 'w';
    positionEditor.fullmove = parseInt(fullmove) || 1;
    positionEditor.pickedSq = null;
}

function editorBuildFen() {
    const rows = [];
    for (let rank = 7; rank >= 0; rank--) {
        let row = '';
        let empty = 0;
        for (let file = 0; file < 8; file++) {
            const p = positionEditor.pieces[rank * 8 + file];
            if (p) {
                if (empty) { row += empty; empty = 0; }
                row += p;
            } else {
                empty++;
            }
        }
        if (empty) row += empty;
        rows.push(row);
    }
    const turn = document.getElementById('editor-turn').value;
    // Droits de roque : le serveur ne garde que ceux encore possibles
    return `${rows.join('/')} ${turn} KQkq - 0 ${positionEditor.fullmove}`;
}

function editorRender() {
    for (let sq = 0; sq < 64; sq++) {
        const el = document.getElementById(`sq-${sq}`);
        if (!el) continue;
        const p = positionEditor.pieces[sq];
        el.innerText = p ? (PIECE_UNICODE[p] || p) : '';
        el.classList.remove('highlight-from', 'highlight-to');
        el.classList.toggle('edit-picked', sq === positionEditor.pickedSq);
    }
    const arrow = document.getElementById('svg-move-arrow');
    if (arrow) arrow.style.display = 'none';
}

function editorSetTool(tool) {
    positionEditor.tool = tool;
    positionEditor.pickedSq = null;
    document.querySelectorAll('#editor-palette .palette-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === tool);
    });
    editorRender();
}

function editorClickSquare(sq) {
    const ed = positionEditor;
    if (ed.tool === 'x') {
        ed.pieces[sq] = null;
    } else if (ed.tool !== 'move') {
        // Re-cliquer avec la même pièce l'enlève
        ed.pieces[sq] = ed.pieces[sq] === ed.tool ? null : ed.tool;
    } else if (ed.pickedSq === null) {
        if (ed.pieces[sq]) ed.pickedSq = sq;
    } else {
        if (sq !== ed.pickedSq) {
            ed.pieces[sq] = ed.pieces[ed.pickedSq];
            ed.pieces[ed.pickedSq] = null;
        }
        ed.pickedSq = null;
    }
    editorRender();
}

function editorOpen() {
    fetch('/api/game/status').then(r => r.json()).then(data => {
        editorLoadFen((data.game && data.game.fen) || START_FEN);
        positionEditor.active = true;
        document.getElementById('position-editor').style.display = 'block';
        document.getElementById('btn-edit-position').classList.add('active');
        document.getElementById('digital-chess-board').classList.add('editing');
        document.getElementById('banner-action').innerText =
            '✏️ Correction de la position : placez les pièces comme sur votre échiquier, puis « Appliquer ».';
        editorSetTool('move');
    });
}

function editorClose() {
    positionEditor.active = false;
    positionEditor.pickedSq = null;
    document.getElementById('position-editor').style.display = 'none';
    document.getElementById('btn-edit-position').classList.remove('active');
    const board = document.getElementById('digital-chess-board');
    board.classList.remove('editing');
    board.querySelectorAll('.edit-picked').forEach(el => el.classList.remove('edit-picked'));
    pollGameStatus();
}

function editorApply() {
    postJSON('/api/game/set-position', { fen: editorBuildFen() }).then(res => {
        if (res.success) {
            editorClose();
            document.getElementById('banner-action').innerText = `✏️ ${res.message}`;
        } else {
            document.getElementById('banner-action').innerText = `⚠️ ${res.message || res.error || 'Position refusée'}`;
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const board = document.getElementById('digital-chess-board');

    document.getElementById('btn-edit-position').addEventListener('click', () => {
        if (positionEditor.active) editorClose(); else editorOpen();
    });

    board.addEventListener('click', e => {
        if (!positionEditor.active) return;
        const sqEl = e.target.closest('.chess-sq');
        if (sqEl) editorClickSquare(parseInt(sqEl.dataset.sq));
    });
    board.addEventListener('contextmenu', e => {
        if (!positionEditor.active) return;
        const sqEl = e.target.closest('.chess-sq');
        if (!sqEl) return;
        e.preventDefault();
        positionEditor.pieces[parseInt(sqEl.dataset.sq)] = null;
        positionEditor.pickedSq = null;
        editorRender();
    });

    document.getElementById('editor-palette').addEventListener('click', e => {
        const btn = e.target.closest('.palette-btn');
        if (btn) editorSetTool(btn.dataset.tool);
    });
    document.getElementById('btn-editor-start').addEventListener('click', () => {
        editorLoadFen(START_FEN);
        editorRender();
    });
    document.getElementById('btn-editor-clear').addEventListener('click', () => {
        positionEditor.pieces.fill(null);
        positionEditor.pickedSq = null;
        editorRender();
    });
    document.getElementById('btn-editor-apply').addEventListener('click', editorApply);
    document.getElementById('btn-editor-cancel').addEventListener('click', editorClose);
});
