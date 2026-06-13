/**
 * websocket.js — Real-Time WebSocket Inference (CNN+BiLSTM Pipeline)
 * 
 * PREDICTION FLOW (Modules 1-3):
 *   1. Frontend sends base64 JPEG frames via WebSocket 'frame' event
 *   2. Backend buffers 20 frames → CNN+BiLSTM inference
 *   3. Backend emits 'prediction_result' with { word, confidence }
 *      - 'word' is present ONLY when confidence > 0.75
 *   4. Frontend receives 'prediction_result':
 *      - Updates UI with the detected English word
 *      - IMMEDIATELY triggers fetch to /api/translate with selected language
 *   5. If 'word' is null → show diagnostic info (low-confidence / buffering)
 * 
 * Depends on: state.js
 */

function initWebSocket() {
    try {
        wsSocket = io({ transports: ['websocket', 'polling'] });

        wsSocket.on('connect', () => {
            console.log('[WS] Connected to server for CNN+BiLSTM inference');
        });
        wsSocket.on('disconnect', () => {
            console.log('[WS] Disconnected');
        });
        wsSocket.on('status', (data) => {
            console.log('[WS] Server status:', data);
            if (data.model === 'cnn_bilstm') {
                console.log('[WS] ✓ CNN+BiLSTM model active (seq_len=' + data.seq_len + ')');
                console.log('[WS] ✓ Confidence threshold: ' + data.confidence_threshold);
            } else {
                console.warn('[WS] ✗ No model loaded on server');
            }
        });

        // ══════════════════════════════════════════════════════════
        // MODULE 3: Listen for 'prediction_result' from backend
        // This is the SOLE prediction event. The backend only sends
        // 'word' when confidence > 0.75.
        // ══════════════════════════════════════════════════════════
        wsSocket.on('prediction_result', (data) => {
            console.log('[WS] prediction_result received:', JSON.stringify(data));
            handlePredictionResult(data);
        });

        // ── CNN+BiLSTM: Buffer status events from server ──
        wsSocket.on('buffer_status', (data) => {
            wsBufferCount = data.count;
            wsBufferRequired = data.required;
            wsBufferReady = data.ready;
            _updateBufferIndicator(data);
        });

        // Create offscreen canvas for frame capture (compressed JPEG for WebSocket)
        frameCaptureCanvas = document.createElement('canvas');
        frameCaptureCanvas.width = FRAME_CAPTURE_WIDTH;
        frameCaptureCanvas.height = FRAME_CAPTURE_HEIGHT;
        frameCaptureCtx = frameCaptureCanvas.getContext('2d');

    } catch (err) {
        console.error('WebSocket init failed:', err);
    }
}


/**
 * MODULE 3 — Frontend Sync: Handle prediction_result from backend
 * 
 * CASE 1: data.word is present → High-confidence detection
 *   - Update UI with the detected English word
 *   - IMMEDIATELY trigger /api/translate with selected target_language
 * 
 * CASE 2: data.word is null → Below threshold / buffering
 *   - Show diagnostic status info
 */
function handlePredictionResult(data) {
    if (isRecording || isProcessing) return;

    const badge = document.getElementById('detectionBadge');

    // ── Always update the diagnostic debug panel ──
    _updateDebugPanel(data);

    // ═══════════════════════════════════════════════════════════════
    // CASE 1: CONFIDENT DETECTION — 'word' is present (conf > 0.75)
    // ═══════════════════════════════════════════════════════════════
    if (data.word) {
        const confPct = Math.round(data.confidence * 100);
        console.log(`[PREDICTION] ✓ Detected: "${data.word}" at ${confPct}% confidence`);

        // Update detection badge in camera overlay
        if (badge) {
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-hand" style="color:#10b981"></i>' +
                '<span>' + data.word + ' (' + confPct + '%)</span>';
        }

        // Update global state
        currentSign = data.word;
        currentConfidence = data.confidence;

        // Update the detected-sign display card
        const signEl = document.getElementById('currentSign');
        if (signEl) signEl.textContent = data.word;

        const confEl = document.getElementById('signsDetail');
        if (confEl) {
            let confClass = confPct >= 85 ? 'conf-high' : confPct >= 60 ? 'conf-mid' : 'conf-low';
            confEl.innerHTML = '<span class="conf-value ' + confClass + '">' + confPct +
                '%</span> confidence &middot; CNN+BiLSTM &middot; Live';
        }

        // Enable the "Add Word" button
        const addBtn = document.getElementById('addWordBtn');
        if (addBtn) addBtn.disabled = false;

        // Update the "English" output panel
        const origEl = document.getElementById('originalText');
        if (origEl) origEl.textContent = data.word;

        // ══════════════════════════════════════════════════════════
        // MODULE 3: IMMEDIATELY trigger /api/translate
        // Only fires when the word is DIFFERENT from the last one
        // (debounce to avoid spamming the API)
        // ══════════════════════════════════════════════════════════
        if (data.word !== lastTranslatedSign && !liveTranslationInFlight) {
            console.log(`[TRANSLATION] Triggering auto-translate: "${data.word}"`);
            _autoTranslateLiveSign(data.word);
        }

        // Reset tracking state
        wsConsecutiveCount = 0;
        wsLastSign = data.word;
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    // CASE 2: LOW CONFIDENCE — 'word' is null
    // ═══════════════════════════════════════════════════════════════
    if (data.raw_sign && data.confidence > 0) {
        const rawPct = Math.round(data.confidence * 100);
        console.log(`[PREDICTION] ✗ Below threshold: "${data.raw_sign}" at ${rawPct}%`);
        if (badge) {
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-magnifying-glass" style="color:#fbbf24"></i>' +
                '<span>' + (data.status || data.raw_sign + ' (' + rawPct + '%)') + '</span>';
        }
    }
}


/**
 * MODULE 3 — Translation Handoff
 * 
 * IMMEDIATELY triggers a fetch call to /api/translate when a confident
 * sign is detected. Uses the currently selected target_language from
 * the UI dropdown (#targetLanguage).
 * 
 * @param {string} word - The detected English word (e.g., "Hello")
 */
async function _autoTranslateLiveSign(word) {
    if (!word || !word.trim()) return;

    // Get the currently selected target language from the dropdown
    const langSelect = document.getElementById('targetLanguage');
    if (!langSelect) {
        console.error('[TRANSLATION] targetLanguage dropdown not found!');
        return;
    }
    const targetLang = langSelect.value;
    console.log(`[TRANSLATION] Calling /api/translate: word="${word}", target_lang="${targetLang}"`);

    liveTranslationInFlight = true;
    lastTranslatedSign = word;

    try {
        const resp = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: word,
                target_lang: targetLang,
                direction: 'en_to_regional'
            })
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[TRANSLATION] HTTP error', resp.status, errData);
            _showTranslationError();
            return;
        }

        const data = await resp.json();
        if (data.success && data.data) {
            const transEl = document.getElementById('translatedText');
            if (transEl) {
                transEl.style.color = '';
                transEl.textContent = data.data.translated;
            }
            console.log(`[TRANSLATION] ✓ "${word}" → "${data.data.translated}" (${targetLang})`);
        } else {
            console.error('[TRANSLATION] Backend error:', data.error || data.message);
            _showTranslationError();
        }
    } catch (err) {
        console.error('[TRANSLATION] Network error:', err);
        _showTranslationError();
    } finally {
        liveTranslationInFlight = false;
    }
}


/**
 * CNN+BiLSTM Frame Sender
 * 
 * Captures the current video frame from the <video> element, compresses it
 * to a base64 JPEG, and sends it to the backend via WebSocket.
 * 
 * @param {HTMLVideoElement} videoElement - The camera preview element
 */
function sendFrameViaWebSocket(videoElement) {
    if (!wsSocket || !wsSocket.connected) return;
    if (isRecording || isProcessing) return;

    const now = Date.now();
    if (now - wsLastSendTime < WS_THROTTLE_MS) return;

    if (!frameCaptureCanvas || !frameCaptureCtx) return;
    if (!videoElement || videoElement.readyState < 2) return;

    // Draw current video frame onto the offscreen canvas (downscaled)
    frameCaptureCtx.drawImage(videoElement, 0, 0, FRAME_CAPTURE_WIDTH, FRAME_CAPTURE_HEIGHT);

    // Convert to base64 JPEG
    const base64Data = frameCaptureCanvas.toDataURL('image/jpeg', FRAME_CAPTURE_QUALITY);

    // Send to backend
    wsSocket.emit('frame', { image: base64Data });
    wsLastSendTime = now;
}


// ═══════════════════════════════════════════════════════════════════════════
// BUFFER FILL INDICATOR  —  Shows how many frames have been collected
// ═══════════════════════════════════════════════════════════════════════════

let _bufferIndicatorEl = null;

/**
 * Create and update the buffer fill indicator in the camera overlay.
 */
function _updateBufferIndicator(data) {
    if (!_bufferIndicatorEl) {
        _bufferIndicatorEl = document.createElement('div');
        _bufferIndicatorEl.id = 'bufferIndicator';
        _bufferIndicatorEl.style.cssText = `
            position: absolute; bottom: 8px; left: 8px; right: 8px;
            z-index: 4; display: flex; align-items: center; gap: 8px;
            background: rgba(0,0,0,0.7); border-radius: 8px; padding: 6px 10px;
            font-family: 'Inter', monospace; font-size: 0.72rem; color: #a1a1aa;
            backdrop-filter: blur(4px);
        `;
        const camContainer = document.getElementById('cameraContainer');
        if (camContainer) camContainer.appendChild(_bufferIndicatorEl);
    }

    const pct = Math.round((data.count / data.required) * 100);
    const barColor = data.ready
        ? 'linear-gradient(90deg, #10b981, #34d399)'
        : 'linear-gradient(90deg, #6366f1, #818cf8)';
    const statusText = data.ready
        ? '🧠 Predicting...'
        : `📹 Buffering ${data.count}/${data.required}`;

    _bufferIndicatorEl.innerHTML = `
        <span style="white-space:nowrap; min-width: 110px;">${statusText}</span>
        <div style="flex:1; height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
            <div style="width:${pct}%; height:100%; background:${barColor}; border-radius:2px; transition: width 0.15s ease;"></div>
        </div>
        <span style="min-width:30px; text-align:right;">${pct}%</span>
    `;
}


// ═══════════════════════════════════════════════════════════════════════════
// REAL-TIME DEBUG PANEL  —  Displays confidence, model, prediction pipeline
// ═══════════════════════════════════════════════════════════════════════════

let _debugPanelEl = null;
let _debugPredictionCount = 0;
let _debugLastUpdate = 0;

/**
 * Create the floating debug panel and inject it into the camera container.
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
                <span class="debug-label">Confidence</span>
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
                <span class="debug-label">Gate (>75%)</span>
                <span class="debug-value" id="debugGateStatus">--</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Model</span>
                <span class="debug-value" id="debugModel">--</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Predictions</span>
                <span class="debug-value" id="debugFrames">0</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Buffer</span>
                <span class="debug-value" id="debugBuffer">${wsBufferCount}/${wsBufferRequired}</span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Pipeline</span>
                <span class="debug-value" id="debugPipeline">
                    <span class="debug-dot pending"></span> Waiting
                </span>
            </div>
            <div class="debug-row">
                <span class="debug-label">Status</span>
                <span class="debug-value" id="debugStatus" style="font-size:0.65rem;">--</span>
            </div>
        </div>
    `;

    const camContainer = document.getElementById('cameraContainer');
    if (camContainer) {
        camContainer.appendChild(panel);
    } else {
        document.body.appendChild(panel);
    }

    _debugPanelEl = panel;
    return panel;
}

/**
 * Update the debug panel with the latest prediction data.
 */
function _updateDebugPanel(data) {
    if (!_debugPanelEl) _createDebugPanel();

    _debugPredictionCount++;
    const now = Date.now();

    // Update FPS indicator
    if (_debugLastUpdate > 0) {
        const dtMs = now - _debugLastUpdate;
        const fps = dtMs > 0 ? (1000 / dtMs).toFixed(1) : '--';
        const fpsEl = document.getElementById('debugFps');
        if (fpsEl) fpsEl.textContent = fps + ' p/s';
    }
    _debugLastUpdate = now;

    // Prediction text
    const predEl = document.getElementById('debugPrediction');
    if (predEl) predEl.textContent = data.word || data.raw_sign || '(none)';

    // Raw confidence
    const rawConf = data.confidence || 0;
    const rawConfEl = document.getElementById('debugRawConf');
    if (rawConfEl) {
        rawConfEl.textContent = rawConf.toFixed(6);
        if (rawConf >= 0.75) rawConfEl.style.color = '#10b981';
        else if (rawConf >= 0.50) rawConfEl.style.color = '#fbbf24';
        else rawConfEl.style.color = '#ef4444';
    }

    // Confidence bar
    const confPct = Math.round(rawConf * 100);
    const confBar = document.getElementById('debugConfBar');
    const confPctEl = document.getElementById('debugConfPct');
    if (confBar) {
        confBar.style.width = confPct + '%';
        if (confPct >= 75) confBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        else if (confPct >= 50) confBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        else confBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
    if (confPctEl) confPctEl.textContent = confPct + '%';

    // Gate status (>75%)
    const gateEl = document.getElementById('debugGateStatus');
    if (gateEl) {
        if (data.word) {
            gateEl.innerHTML = '<span style="color:#10b981">✓ PASS</span>';
        } else if (rawConf >= 0.75) {
            gateEl.innerHTML = '<span style="color:#fbbf24">⏳ Stabilising</span>';
        } else {
            gateEl.innerHTML = '<span style="color:#ef4444">✗ BELOW</span>';
        }
    }

    // Model type
    const modelEl = document.getElementById('debugModel');
    if (modelEl) {
        modelEl.textContent = 'cnn_bilstm';
        modelEl.style.color = '#f472b6';
    }

    // Prediction counter
    const framesEl = document.getElementById('debugFrames');
    if (framesEl) framesEl.textContent = _debugPredictionCount;

    // Buffer status
    const bufferEl = document.getElementById('debugBuffer');
    if (bufferEl) bufferEl.textContent = `${wsBufferCount}/${wsBufferRequired}`;

    // Pipeline health
    const pipeEl = document.getElementById('debugPipeline');
    if (pipeEl) {
        if (data.word) {
            pipeEl.innerHTML = '<span class="debug-dot healthy"></span> Detected ✓';
        } else if (rawConf >= 0.75) {
            pipeEl.innerHTML = '<span class="debug-dot warning"></span> Close';
        } else if (rawConf >= 0.50) {
            pipeEl.innerHTML = '<span class="debug-dot warning"></span> Low Conf';
        } else if (data.raw_sign) {
            pipeEl.innerHTML = '<span class="debug-dot danger"></span> Below Gate';
        } else {
            pipeEl.innerHTML = '<span class="debug-dot pending"></span> Buffering';
        }
    }

    // Status message
    const statusEl = document.getElementById('debugStatus');
    if (statusEl) {
        statusEl.textContent = data.status || (data.word ? 'Detected: ' + data.word : '--');
    }

    console.log(
        `[DIAG] #${_debugPredictionCount} | ` +
        `word="${data.word}" raw="${data.raw_sign || ''}" | conf=${rawConf?.toFixed(6)} | ` +
        `status="${data.status || ''}"`
    );
}
