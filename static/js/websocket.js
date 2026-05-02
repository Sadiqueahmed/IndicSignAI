/**
 * websocket.js — Real-Time WebSocket Inference
 * Depends on: state.js
 */

function initWebSocket() {
    try {
        wsSocket = io({ transports: ['websocket', 'polling'] });

        wsSocket.on('connect', () => {
            console.log('[WS] Connected to server for real-time inference');
        });
        wsSocket.on('disconnect', () => {
            console.log('[WS] Disconnected');
        });
        wsSocket.on('status', (data) => {
            console.log('[WS] Server status:', data);
            if (data.sklearn) console.log('[WS] Sklearn model active');
            if (data.tflite) console.log('[WS] TFLite acceleration active');
        });
        wsSocket.on('prediction', (data) => {
            handleWebSocketPrediction(data);
        });

        // Offscreen canvas for full-frame capture (backend runs MediaPipe)
        wsOffCanvas = document.createElement('canvas');
        wsOffCanvas.width = 320;
        wsOffCanvas.height = 240;
        wsOffCtx = wsOffCanvas.getContext('2d');
    } catch (err) {
        console.error('WebSocket init failed:', err);
    }
}

function handleWebSocketPrediction(data) {
    if (isRecording || isProcessing) return;

    const badge = document.getElementById('detectionBadge');

    // ── DIAGNOSTIC: Always update the debug panel with every prediction ──
    _updateDebugPanel(data);

    if (data.sign && data.confidence >= 0.45) {
        if (data.sign === wsLastSign) {
            wsConsecutiveCount++;
        } else {
            wsLastSign = data.sign;
            wsConsecutiveCount = 1;
        }

        if (wsConsecutiveCount >= WS_CONSECUTIVE_LOCK) {
            const confPct = Math.round(data.confidence * 100);

            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-hand" style="color:#10b981"></i>' +
                '<span>' + data.sign + ' (' + confPct + '%)</span>';

            currentSign = data.sign;
            currentConfidence = data.confidence;
            document.getElementById('currentSign').textContent = data.sign;

            const confEl = document.getElementById('signsDetail');
            let confClass = confPct >= 70 ? 'conf-high' : confPct >= 40 ? 'conf-mid' : 'conf-low';
            confEl.innerHTML = '<span class="conf-value ' + confClass + '">' + confPct +
                '%</span> confidence &middot; ' + data.model + ' &middot; Live';

            document.getElementById('addWordBtn').disabled = false;

            // ── AUTO-TRANSLATE: debounced live translation ──────────────
            // Only fire if this is a genuinely NEW sign and no request is
            // already in-flight.  Captures the dropdown value at the exact
            // moment the sign is confirmed.
            if (data.sign !== lastTranslatedSign && !liveTranslationInFlight) {
                _autoTranslateLiveSign(data.sign);
            }
        }
    } else {
        wsConsecutiveCount = Math.max(0, wsConsecutiveCount - 1);
    }
}

/**
 * Debounced auto-translation for the live camera pipeline.
 * - Reads the language dropdown value at call time.
 * - Guards against empty/null signs.
 * - Sets liveTranslationInFlight to prevent concurrent calls.
 * - On failure, shows the red toast + fallback text via _showTranslationError().
 */
async function _autoTranslateLiveSign(sign) {
    if (!sign || !sign.trim()) return;

    const langSelect = document.getElementById('targetLanguage');
    if (!langSelect) return;
    const targetLang = langSelect.value;

    liveTranslationInFlight = true;
    lastTranslatedSign = sign;

    // Update the English text display immediately
    const origEl = document.getElementById('originalText');
    if (origEl) origEl.textContent = sign;

    try {
        const resp = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: sign,
                target_lang: targetLang,
                direction: 'en_to_regional'
            })
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[_autoTranslateLiveSign] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }

        const data = await resp.json();
        if (data.success && data.data) {
            const transEl = document.getElementById('translatedText');
            if (transEl) {
                transEl.style.color = '';  // reset any error styling
                transEl.textContent = data.data.translated;
            }
        } else {
            console.error('[_autoTranslateLiveSign] Backend error:', data.error || data.message);
            _showTranslationError();
        }
    } catch (err) {
        console.error('[_autoTranslateLiveSign] Network error:', err);
        _showTranslationError();
    } finally {
        liveTranslationInFlight = false;
    }
}

/**
 * Send the FULL video frame to the backend via WebSocket.
 * 
 * The backend runs MediaPipe + landmark extraction + model prediction.
 * We send the full frame (not a cropped region) because:
 *   1. MediaPipe needs the full image context for accurate hand detection
 *   2. The backend handles bounding-box normalisation identically to training
 *   3. Pre-cropping a 160×160 region caused MediaPipe re-detection failures
 */
function sendFrameViaWebSocket(landmarks, videoElement) {
    if (!wsSocket || !wsSocket.connected) return;
    if (isRecording || isProcessing) return;

    const now = Date.now();
    if (now - wsLastSendTime < WS_THROTTLE_MS) return;

    // Motion blur detection — skip frames with too much hand movement
    if (wsPrevLandmarks && wsPrevLandmarks.length === landmarks.length) {
        let totalDisplacement = 0;
        for (let i = 0; i < landmarks.length; i++) {
            const dx = landmarks[i].x - wsPrevLandmarks[i].x;
            const dy = landmarks[i].y - wsPrevLandmarks[i].y;
            totalDisplacement += Math.sqrt(dx * dx + dy * dy);
        }
        const avgMotion = totalDisplacement / landmarks.length;
        if (avgMotion > WS_MOTION_THRESHOLD) {
            const badge = document.getElementById('detectionBadge');
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-wind" style="color:#f59e0b"></i><span>Hold steady...</span>';
            wsPrevLandmarks = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
            return;
        }
    }
    wsPrevLandmarks = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));

    // Send only the extracted landmarks to the backend for inference,
    // bypassing the need to send Base64 images and run MediaPipe twice.
    const lmArray = landmarks.map(lm => ({
        x: lm.x,
        y: lm.y,
        z: lm.z || 0.0
    }));
    
    wsSocket.emit('frame', { landmarks: lmArray });
    wsLastSendTime = now;
}

// ═══════════════════════════════════════════════════════════════════════════
// REAL-TIME DEBUG PANEL  —  Displays confidence, model, prediction pipeline
// ═══════════════════════════════════════════════════════════════════════════

let _debugPanelEl = null;
let _debugPredictionCount = 0;
let _debugLastUpdate = 0;

/**
 * Create the floating debug panel and inject it into the camera container.
 * The panel appears as a semi-transparent overlay in the bottom-left corner
 * of the camera feed — always visible while the camera is active.
 */
function _createDebugPanel() {
    if (_debugPanelEl) return _debugPanelEl;

    const panel = document.createElement('div');
    panel.id = 'debugPanel';
    panel.innerHTML = `
        <div class="debug-header">
            <i class="fa-solid fa-bug"></i>
            <span>DIAGNOSTIC PANEL</span>
            <span class="debug-fps" id="debugFps">--</span>
        </div>
        <div class="debug-rows">
            <div class="debug-row">
                <span class="debug-label">Prediction</span>
                <span class="debug-value" id="debugPrediction">--</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Raw Confidence</span>
                <span class="debug-value" id="debugRawConf">--</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Confidence %</span>
                <div class="debug-conf-bar-wrap">
                    <div class="debug-conf-bar" id="debugConfBar" style="width: 0%"></div>
                </div>
                <span class="debug-value debug-pct" id="debugConfPct">0%</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Model</span>
                <span class="debug-value" id="debugModel">--</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Frames</span>
                <span class="debug-value" id="debugFrames">0</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Pipeline</span>
                <span class="debug-value" id="debugPipeline">
                    <span class="debug-dot pending"></span> Waiting
                </span>
            </div>
        </div>
    `;

    // Inject into the camera container (z-index above canvas but below processing overlay)
    const camContainer = document.getElementById('cameraContainer');
    if (camContainer) {
        camContainer.appendChild(panel);
    } else {
        // Fallback: append to body
        document.body.appendChild(panel);
    }

    _debugPanelEl = panel;
    return panel;
}

/**
 * Update the debug panel with the latest prediction data.
 * Called on EVERY WebSocket prediction — even low-confidence ones.
 */
function _updateDebugPanel(data) {
    if (!_debugPanelEl) _createDebugPanel();

    _debugPredictionCount++;
    const now = Date.now();

    // Update FPS indicator (predictions per second)
    if (_debugLastUpdate > 0) {
        const dtMs = now - _debugLastUpdate;
        const fps = dtMs > 0 ? (1000 / dtMs).toFixed(1) : '--';
        const fpsEl = document.getElementById('debugFps');
        if (fpsEl) fpsEl.textContent = fps + ' p/s';
    }
    _debugLastUpdate = now;

    // Prediction text
    const predEl = document.getElementById('debugPrediction');
    if (predEl) predEl.textContent = data.sign || '(none)';

    // Raw confidence float (the key diagnostic value)
    const rawConf = data.raw_confidence !== undefined ? data.raw_confidence : data.confidence;
    const rawConfEl = document.getElementById('debugRawConf');
    if (rawConfEl) {
        rawConfEl.textContent = rawConf !== undefined ? rawConf.toFixed(6) : '--';
        // Color code: green > 0.75, yellow 0.45-0.75, red < 0.45
        if (rawConf >= 0.75) rawConfEl.style.color = '#10b981';
        else if (rawConf >= 0.45) rawConfEl.style.color = '#fbbf24';
        else rawConfEl.style.color = '#ef4444';
    }

    // Confidence bar
    const confPct = rawConf !== undefined ? Math.round(rawConf * 100) : 0;
    const confBar = document.getElementById('debugConfBar');
    const confPctEl = document.getElementById('debugConfPct');
    if (confBar) {
        confBar.style.width = confPct + '%';
        if (confPct >= 75) confBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        else if (confPct >= 45) confBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        else confBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
    if (confPctEl) confPctEl.textContent = confPct + '%';

    // Model type
    const modelEl = document.getElementById('debugModel');
    if (modelEl) {
        modelEl.textContent = data.model || '--';
        modelEl.style.color = data.model === 'sklearn' ? '#06b6d4' :
                              data.model === 'tflite' ? '#a78bfa' :
                              data.model === 'keras' ? '#f472b6' : 'inherit';
    }

    // Frame counter
    const framesEl = document.getElementById('debugFrames');
    if (framesEl) framesEl.textContent = _debugPredictionCount;

    // Pipeline health
    const pipeEl = document.getElementById('debugPipeline');
    if (pipeEl) {
        if (rawConf >= 0.75) {
            pipeEl.innerHTML = '<span class="debug-dot healthy"></span> Healthy';
        } else if (rawConf >= 0.45) {
            pipeEl.innerHTML = '<span class="debug-dot warning"></span> Low Conf';
        } else if (data.sign) {
            pipeEl.innerHTML = '<span class="debug-dot danger"></span> Garbage Data';
        } else {
            pipeEl.innerHTML = '<span class="debug-dot pending"></span> No Prediction';
        }
    }

    // Console log for cross-referencing with backend
    console.log(
        `[DIAG Panel] #${_debugPredictionCount} | ` +
        `sign="${data.sign}" | raw_conf=${rawConf?.toFixed(6)} | ` +
        `model=${data.model}`
    );
}
