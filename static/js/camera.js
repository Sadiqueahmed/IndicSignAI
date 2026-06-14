/**
 * camera.js — MediaPipe Hands + WebRTC Camera Lifecycle
 * 
 * ARCHITECTURE (ISL_IMAGE):
 * MediaPipe Hands runs client-side for VISUAL FEEDBACK
 * (skeleton drawing, bounding box, fingertip highlights)
 * AND for providing hand landmarks to the backend.
 * 
 * We send the RAW camera frame + landmarks via sendFrameViaWebSocket().
 * The backend crops the hand and runs the ISL_IMAGE model (MobileNetV2 + Transformer).
 * 
 * Frame sending is gated by `isDetectionActive` — the user must click
 * "Start Detection" before any frames are streamed to the server.
 * 
 * Depends on: state.js, websocket.js (sendFrameViaWebSocket)
 */

function initMediaPipeHands(videoElement) {
    const canvas = document.getElementById('handCanvas');
    if (!canvas) { console.error('handCanvas not found'); return; }
    handCanvasCtx = canvas.getContext('2d');

    function syncCanvasSize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    syncCanvasSize();
    window.addEventListener('resize', syncCanvasSize);
    videoElement.addEventListener('resize', syncCanvasSize);

    try {
        mpHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
        });
        mpHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.8,
            minTrackingConfidence: 0.8
        });
        mpHands.onResults((results) => onMediaPipeResults(results, canvas, handCanvasCtx, videoElement));

        mpCamera = new Camera(videoElement, {
            onFrame: async () => {
                if (mpHands && !isProcessing) await mpHands.send({ image: videoElement });
            },
            width: 640, height: 480
        });
        mpCamera.start();
        console.log('✔ MediaPipe Hands initialized (ISL_IMAGE pipeline)');
    } catch (err) {
        console.error('MediaPipe Hands init failed:', err);
    }
}

function onMediaPipeResults(results, canvas, ctx, videoElement) {
    if (isProcessing) return;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── Track whether hands are visible (for badge updates) ──
    const handsDetected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    if (handsDetected) {
        if (isDetectionActive) {
            const badge = document.getElementById('detectionBadge');
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-hand" style="color:#10b981"></i><span>Hand detected — streaming</span>';
        }

        for (const landmarks of results.multiHandLandmarks) {
            // Draw hand skeleton connections
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: 'rgba(99,102,241,0.7)', lineWidth: 3 });
            // Draw landmark points
            drawLandmarks(ctx, landmarks, { color: '#10b981', lineWidth: 1, radius: 4 });

            // Highlight fingertips (indices 4,8,12,16,20)
            for (const idx of [4, 8, 12, 16, 20]) {
                const lm = landmarks[idx];
                ctx.beginPath();
                ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 7, 0, 2 * Math.PI);
                ctx.fillStyle = '#ec4899'; ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2; ctx.stroke();
            }

            // ── VISUAL DEBUGGING: Draw bounding box around the hand ──
            let xMin = 1, xMax = 0, yMin = 1, yMax = 0;
            for (const lm of landmarks) {
                if (lm.x < xMin) xMin = lm.x;
                if (lm.x > xMax) xMax = lm.x;
                if (lm.y < yMin) yMin = lm.y;
                if (lm.y > yMax) yMax = lm.y;
            }
            const pad = 0.03;
            const bx = Math.max(0, xMin - pad) * canvas.width;
            const by = Math.max(0, yMin - pad) * canvas.height;
            const bw = Math.min(1, xMax + pad) * canvas.width - bx;
            const bh = Math.min(1, yMax + pad) * canvas.height - by;

            ctx.strokeStyle = 'rgba(250, 204, 21, 0.7)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.setLineDash([]);
        }
    } else if (!isProcessing) {
        const badge = document.getElementById('detectionBadge');
        badge.classList.add('active');
        if (isDetectionActive) {
            badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Detection active — show your hand</span>';
        } else {
            badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Camera ready — tap Start Detection</span>';
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // ISL_IMAGE: Send RAW video frame + landmarks to backend
    // The backend uses landmarks to crop the hand region, then runs
    // the ISL_IMAGE model (MobileNetV2 + Transformer) on the crop.
    // We only send when hands ARE detected since landmarks are required.
    // ═══════════════════════════════════════════════════════════════
    if (handsDetected && videoElement) {
        // Pass the first detected hand's landmarks to the WebSocket sender
        const firstHandLandmarks = results.multiHandLandmarks[0];
        sendFrameViaWebSocket(videoElement, firstHandLandmarks);
    }

    ctx.restore();
}

async function startCamera() {
    const video = document.getElementById('cameraPreview');
    const badge = document.getElementById('detectionBadge');

    // ── LIFECYCLE CLEANUP: destroy any existing instances first ──
    if (mpCamera) { try { mpCamera.stop(); } catch (e) {} mpCamera = null; }
    if (mpHands)  { try { mpHands.close(); } catch (e) {} mpHands  = null; }
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    if (video) video.srcObject = null;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        video.srcObject = cameraStream;
        badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Camera ready — show your hand</span>';
        badge.classList.add('active');
        video.addEventListener('loadeddata', () => initMediaPipeHands(video), { once: true });
    } catch (err) {
        console.error('Camera access denied:', err);
        badge.innerHTML = '<i class="fa-solid fa-exclamation-triangle"></i><span>Camera access denied</span>';
        showToast('Camera access denied.', 'error');
    }
}

function stopCamera() {
    if (mpCamera) { try { mpCamera.stop(); } catch (e) {} mpCamera = null; }
    if (mpHands)  { try { mpHands.close(); } catch (e) {} mpHands  = null; }
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    const video = document.getElementById('cameraPreview');
    if (video) video.srcObject = null;
}
