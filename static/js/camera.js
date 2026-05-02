/**
 * camera.js — MediaPipe Hands + WebRTC Camera Lifecycle
 * Depends on: state.js, websocket.js (sendFrameViaWebSocket)
 */

function initMediaPipeHands(videoElement) {
    const canvas = document.getElementById('handCanvas');
    if (!canvas) { console.error('handCanvas not found'); return; }
    handCanvasCtx = canvas.getContext('2d');

    function syncCanvasSize() {
        // Sync to parent container AND to actual video dimensions
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }
    syncCanvasSize();
    window.addEventListener('resize', syncCanvasSize);
    // Also sync when the video metadata loads (actual video dimensions available)
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
        mpHands.onResults((results) => onMediaPipeResults(results, canvas, handCanvasCtx));

        mpCamera = new Camera(videoElement, {
            onFrame: async () => {
                if (mpHands && !isProcessing) await mpHands.send({ image: videoElement });
            },
            width: 640, height: 480
        });
        mpCamera.start();
        console.log('✔ MediaPipe Hands initialized');
    } catch (err) {
        console.error('MediaPipe Hands init failed:', err);
    }
}

function onMediaPipeResults(results, canvas, ctx) {
    if (isProcessing) return;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        if (!isRecording) {
            const badge = document.getElementById('detectionBadge');
            badge.classList.add('active');
            badge.innerHTML = '<i class="fa-solid fa-hand" style="color:#10b981"></i><span>Hand detected  -  tap record</span>';
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
            // This confirms that the frontend sees the hand correctly.
            // If the bbox doesn't track the hand, it's a lighting/camera issue.
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

            ctx.strokeStyle = 'rgba(250, 204, 21, 0.7)';  // Yellow bbox
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(bx, by, bw, bh);
            ctx.setLineDash([]);
        }

        // Send FULL frame to backend for inference
        const videoEl = document.getElementById('cameraPreview');
        if (videoEl && results.multiHandLandmarks[0]) {
            // ── DIAGNOSTIC: Log landmark array shape before sending ──
            const lm = results.multiHandLandmarks[0];
            console.log(
                `[DIAG] Frontend landmarks: ${results.multiHandLandmarks.length} hand(s), ` +
                `${lm.length} points, first=(${lm[0].x.toFixed(4)}, ${lm[0].y.toFixed(4)}, ${lm[0].z.toFixed(4)})`
            );
            sendFrameViaWebSocket(lm, videoEl);
        }
    } else if (!isRecording && !isProcessing) {
        const badge = document.getElementById('detectionBadge');
        badge.classList.add('active');
        badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Camera ready  -  tap record</span>';
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
        badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Camera ready  -  tap record</span>';
        badge.classList.add('active');
        // { once: true } ensures initMediaPipeHands fires exactly once per camera start
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
    if (isRecording) stopRecording();
}
