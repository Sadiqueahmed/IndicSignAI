/**
 * websocket.js — Real-Time WebSocket Inference (ISL_IMAGE Pipeline)
 * 
 * PREDICTION FLOW:
 *   1. Frontend captures base64 JPEG frame + MediaPipe landmarks
 *   2. Sends both via WebSocket 'frame' event
 *   3. Backend crops hand from image using landmarks → ISL_IMAGE model (160×160)
 *   4. Backend emits 'prediction' with { sign, confidence }
 *   5. Frontend accumulates signs into liveSentenceBuffer
 *   6. After 2s of silence, sends entire buffer to /api/correct-and-translate
 *   7. Grammar-corrected + translated result displayed in UI
 * 
 * Depends on: state.js
 */

function initWebSocket() {
    try {
        wsSocket = io({ transports: ['websocket', 'polling'] });

        wsSocket.on('connect', () => {
            console.log('[WS] Connected to server for ISL_IMAGE inference');
        });
        wsSocket.on('disconnect', () => {
            console.log('[WS] Disconnected');
        });
        wsSocket.on('status', (data) => {
            console.log('[WS] Server status:', data);
            if (data.model === 'isl_image') {
                console.log('[WS] ✓ ISL_IMAGE model active (MobileNetV2 + Transformer)');
                console.log('[WS] ✓ Confidence threshold: ' + data.confidence_threshold);
            } else {
                console.warn('[WS] ✗ No model loaded on server');
            }
        });

        // ══════════════════════════════════════════════════════════
        // Listen for 'prediction' from backend (ISL_IMAGE model)
        // ══════════════════════════════════════════════════════════
        wsSocket.on('prediction', (data) => {
            console.log('[WS] prediction received:', JSON.stringify(data));
            handleWebSocketPrediction(data);
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
 * Handle prediction from backend (ISL_IMAGE model)
 * 
 * CASE 1: data.sign is present → High-confidence detection
 *   - Update UI with the detected sign
 *   - Append to liveSentenceBuffer (avoid consecutive duplicates)
 *   - Reset the grammar debounce timer (2s)
 * 
 * CASE 2: data.sign is null → Below threshold / error
 *   - Show diagnostic status info
 */
function handleWebSocketPrediction(data) {
    if (!isDetectionActive || isProcessing) return;

    const badge = document.getElementById('detectionBadge');

    // ── Always update the diagnostic debug panel ──
    _updateDebugPanel(data);

    // ═══════════════════════════════════════════════════════════════
    // CASE 1: CONFIDENT DETECTION — 'sign' is present (conf > 0.65)
    // ═══════════════════════════════════════════════════════════════
    if (data.sign) {
        const confPct = Math.round(data.confidence * 100);
        console.log(`[PREDICTION] ✓ Detected: "${data.sign}" at ${confPct}% confidence`);

        // Update detection badge in camera overlay
        if (badge) {
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-hand" style="color:#10b981"></i>' +
                '<span>' + data.sign + ' (' + confPct + '%)</span>';
        }

        // Update global state
        currentSign = data.sign;
        currentConfidence = data.confidence;

        // Update the detected-sign display card
        const signEl = document.getElementById('currentSign');
        if (signEl) signEl.textContent = data.sign;

        const confEl = document.getElementById('signsDetail');
        if (confEl) {
            let confClass = confPct >= 85 ? 'conf-high' : confPct >= 60 ? 'conf-mid' : 'conf-low';
            confEl.innerHTML = '<span class="conf-value ' + confClass + '">' + confPct +
                '%</span> confidence &middot; ISL_IMAGE &middot; Live';
        }

        // Enable the "Add Word" button
        const addBtn = document.getElementById('addWordBtn');
        if (addBtn) addBtn.disabled = false;

        // ══════════════════════════════════════════════════════════
        // SENTENCE BUFFER: Append sign (avoid consecutive duplicates)
        // ══════════════════════════════════════════════════════════
        if (data.sign !== lastBufferedSign) {
            liveSentenceBuffer.push(data.sign);
            lastBufferedSign = data.sign;
            console.log(`[BUFFER] Added "${data.sign}" → buffer: [${liveSentenceBuffer.join(', ')}]`);

            // Update the live sentence buffer display
            _updateSentenceBufferUI();
        }

        // Update the "English" output panel with raw buffer contents
        const origEl = document.getElementById('originalText');
        if (origEl) origEl.textContent = liveSentenceBuffer.join(' ');

        // ══════════════════════════════════════════════════════════
        // GRAMMAR DEBOUNCE: Reset the 2-second timer
        // When the timer fires, send the full buffer for grammar
        // correction + translation (NOT word-by-word).
        // ══════════════════════════════════════════════════════════
        _resetGrammarDebounce();

        // Reset tracking state
        wsConsecutiveCount = 0;
        wsLastSign = data.sign;
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    // CASE 2: LOW CONFIDENCE — 'sign' is null
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
 * GRAMMAR DEBOUNCE — Triggers grammar correction + translation
 * after 2 seconds of no new sign detections.
 * 
 * This ensures we build a FULL sentence before translating,
 * instead of translating word-by-word.
 */
function _resetGrammarDebounce() {
    // Clear any existing timer
    if (grammarDebounceTimer) {
        clearTimeout(grammarDebounceTimer);
        grammarDebounceTimer = null;
    }

    // Set new timer — fires after GRAMMAR_DEBOUNCE_MS (2000ms) of silence
    grammarDebounceTimer = setTimeout(() => {
        grammarDebounceTimer = null;
        _triggerGrammarAndTranslation();
    }, GRAMMAR_DEBOUNCE_MS);
}


/**
 * Send the accumulated sentence buffer to /api/correct-and-translate
 * for grammar correction and native language translation.
 * 
 * The endpoint accepts: { words: ['I', 'GO', 'HOME'], target_lang: 'manipuri' }
 * and returns:          { corrected: 'I am going home.', translated: 'ꯑꯩ ꯌꯨꯝ ꯆꯠꯂꯤ.' }
 */
async function _triggerGrammarAndTranslation() {
    if (liveSentenceBuffer.length === 0) return;
    if (grammarTranslationInFlight) return;

    // Get the target language from the dropdown
    const langSelect = document.getElementById('targetLanguage');
    if (!langSelect) {
        console.error('[TRANSLATION] targetLanguage dropdown not found!');
        return;
    }
    const targetLang = langSelect.value;

    console.log(`[GRAMMAR] Triggering correct-and-translate for buffer: [${liveSentenceBuffer.join(', ')}]`);
    console.log(`[GRAMMAR] Target language: ${targetLang}`);

    grammarTranslationInFlight = true;

    try {
        const resp = await fetch('/api/correct-and-translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                words: [...liveSentenceBuffer],  // copy to avoid mutation
                target_lang: targetLang
            })
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[GRAMMAR] HTTP error', resp.status, errData);
            _showTranslationError();
            return;
        }

        const result = await resp.json();
        if (result.success && result.data) {
            const corrected = result.data.corrected;
            const translated = result.data.translated;

            console.log(`[GRAMMAR] ✓ Corrected: "${corrected}"`);
            console.log(`[GRAMMAR] ✓ Translated: "${translated}" (${targetLang})`);

            // Update the "English" panel with grammar-corrected sentence
            const origEl = document.getElementById('originalText');
            if (origEl) origEl.textContent = corrected;

            // Update the "Translated" panel with native translation
            const transEl = document.getElementById('translatedText');
            if (transEl) {
                transEl.style.color = '';
                transEl.textContent = translated;
            }

            // Update the sentence buffer display to show it's been processed
            const bufferEl = document.getElementById('liveSentenceBufferDisplay');
            if (bufferEl) {
                bufferEl.innerHTML =
                    '<span style="color:#10b981;">✓ </span>' +
                    '<span style="color:#d1d5db;">' + corrected + '</span>';
            }

            // Also populate the Sentence Builder textarea for manual editing
            const sentenceTextEl = document.getElementById('sentenceText');
            if (sentenceTextEl) {
                sentenceTextEl.value = corrected;
            }
            const wordCountEl = document.getElementById('wordCount');
            if (wordCountEl) {
                const wc = corrected.split(/\s+/).filter(w => w).length;
                wordCountEl.textContent = wc + ' word' + (wc !== 1 ? 's' : '');
            }
        } else {
            console.error('[GRAMMAR] Backend error:', result.error || result.message);
            _showTranslationError();
        }
    } catch (err) {
        console.error('[GRAMMAR] Network error:', err);
        _showTranslationError();
    } finally {
        grammarTranslationInFlight = false;
        // Clear the buffer after successful translation
        liveSentenceBuffer = [];
        lastBufferedSign = null;
    }
}


/**
 * Update the live sentence buffer display in the UI.
 * Shows the raw accumulated signs before grammar correction.
 */
function _updateSentenceBufferUI() {
    const bufferEl = document.getElementById('liveSentenceBufferDisplay');
    if (!bufferEl) return;

    if (liveSentenceBuffer.length === 0) {
        bufferEl.innerHTML = '<span style="color:#6b7280; font-style:italic;">Waiting for signs...</span>';
    } else {
        const pills = liveSentenceBuffer.map(w =>
            '<span style="display:inline-block; background:rgba(99,102,241,0.2); ' +
            'border:1px solid rgba(99,102,241,0.4); border-radius:6px; padding:2px 8px; ' +
            'margin:2px; font-size:0.8rem; color:#a5b4fc;">' + w + '</span>'
        ).join(' ');
        bufferEl.innerHTML = pills;
    }
}


/**
 * Clear the live sentence buffer (called from UI "Clear" button or programmatically).
 */
function clearLiveSentenceBuffer() {
    liveSentenceBuffer = [];
    lastBufferedSign = null;
    if (grammarDebounceTimer) {
        clearTimeout(grammarDebounceTimer);
        grammarDebounceTimer = null;
    }
    _updateSentenceBufferUI();

    const origEl = document.getElementById('originalText');
    if (origEl) origEl.textContent = '--';
    const transEl = document.getElementById('translatedText');
    if (transEl) transEl.textContent = '--';
}


/**
 * Show a translation error in the UI.
 */
function _showTranslationError() {
    const transEl = document.getElementById('translatedText');
    if (transEl) {
        transEl.style.color = '#ef4444';
        transEl.textContent = 'Translation failed — check console';
    }
}


/**
 * ISL_IMAGE Frame Sender
 * 
 * Captures the current video frame from the <video> element, compresses it
 * to a base64 JPEG (quality 0.5 for bandwidth), and sends it along with
 * the current MediaPipe hand landmarks to the backend via WebSocket.
 * 
 * @param {HTMLVideoElement} videoElement - The camera preview element
 * @param {Array} landmarks - MediaPipe hand landmarks (21 points with x,y,z)
 */
function sendFrameViaWebSocket(videoElement, landmarks) {
    // ── MASTER GATEKEEPER: Only stream when detection is active ──
    if (!isDetectionActive) return;
    if (!wsSocket || !wsSocket.connected) return;
    if (isProcessing) return;

    const now = Date.now();
    if (now - wsLastSendTime < WS_THROTTLE_MS) return;

    if (!frameCaptureCanvas || !frameCaptureCtx) return;
    if (!videoElement || videoElement.readyState < 2) return;

    // Must have landmarks to crop hand on backend
    if (!landmarks || landmarks.length === 0) return;

    // Draw current video frame onto the offscreen canvas (downscaled)
    frameCaptureCtx.drawImage(videoElement, 0, 0, FRAME_CAPTURE_WIDTH, FRAME_CAPTURE_HEIGHT);

    // Convert to base64 JPEG with high compression (quality 0.5)
    const base64Data = frameCaptureCanvas.toDataURL('image/jpeg', 0.5);

    // Convert MediaPipe landmarks to serializable array of {x, y, z}
    const lmArray = landmarks.map(lm => ({
        x: lm.x,
        y: lm.y,
        z: lm.z || 0
    }));

    // Send both image AND landmarks to backend
    wsSocket.emit('frame', {
        image: base64Data,
        landmarks: lmArray
    });
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
                <span class="debug-label">Gate (>65%)</span>
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
                <span class="debug-value" id="debugBuffer">0 signs</span>
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
    if (predEl) predEl.textContent = data.sign || data.raw_sign || '(none)';

    // Raw confidence
    const rawConf = data.confidence || 0;
    const rawConfEl = document.getElementById('debugRawConf');
    if (rawConfEl) {
        rawConfEl.textContent = rawConf.toFixed(6);
        if (rawConf >= 0.65) rawConfEl.style.color = '#10b981';
        else if (rawConf >= 0.40) rawConfEl.style.color = '#fbbf24';
        else rawConfEl.style.color = '#ef4444';
    }

    // Confidence bar
    const confPct = Math.round(rawConf * 100);
    const confBar = document.getElementById('debugConfBar');
    const confPctEl = document.getElementById('debugConfPct');
    if (confBar) {
        confBar.style.width = confPct + '%';
        if (confPct >= 65) confBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
        else if (confPct >= 40) confBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        else confBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
    if (confPctEl) confPctEl.textContent = confPct + '%';

    // Gate status (>65%)
    const gateEl = document.getElementById('debugGateStatus');
    if (gateEl) {
        if (data.sign) {
            gateEl.innerHTML = '<span style="color:#10b981">✓ PASS</span>';
        } else if (rawConf >= 0.65) {
            gateEl.innerHTML = '<span style="color:#fbbf24">⏳ Stabilising</span>';
        } else {
            gateEl.innerHTML = '<span style="color:#ef4444">✗ BELOW</span>';
        }
    }

    // Model type
    const modelEl = document.getElementById('debugModel');
    if (modelEl) {
        modelEl.textContent = 'isl_image';
        modelEl.style.color = '#f472b6';
    }

    // Prediction counter
    const framesEl = document.getElementById('debugFrames');
    if (framesEl) framesEl.textContent = _debugPredictionCount;

    // Buffer status — show sentence buffer count
    const bufferEl = document.getElementById('debugBuffer');
    if (bufferEl) bufferEl.textContent = `${liveSentenceBuffer.length} signs`;

    // Pipeline health
    const pipeEl = document.getElementById('debugPipeline');
    if (pipeEl) {
        if (data.sign) {
            pipeEl.innerHTML = '<span class="debug-dot healthy"></span> Detected ✓';
        } else if (rawConf >= 0.65) {
            pipeEl.innerHTML = '<span class="debug-dot warning"></span> Close';
        } else if (rawConf >= 0.40) {
            pipeEl.innerHTML = '<span class="debug-dot warning"></span> Low Conf';
        } else if (data.raw_sign) {
            pipeEl.innerHTML = '<span class="debug-dot danger"></span> Below Gate';
        } else {
            pipeEl.innerHTML = '<span class="debug-dot pending"></span> Waiting';
        }
    }

    // Status message
    const statusEl = document.getElementById('debugStatus');
    if (statusEl) {
        statusEl.textContent = data.status || (data.sign ? 'Detected: ' + data.sign : '--');
    }

    console.log(
        `[DIAG] #${_debugPredictionCount} | ` +
        `sign="${data.sign}" raw="${data.raw_sign || ''}" | conf=${rawConf?.toFixed(6)} | ` +
        `buffer=[${liveSentenceBuffer.join(',')}] | ` +
        `status="${data.status || ''}"`
    );
}
