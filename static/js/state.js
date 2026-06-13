/**
 * state.js — Shared Application State
 * All global variables live here so every module can read/write them.
 */

// Current detected sign
let currentSign = null;
let currentConfidence = 0;
let sentenceWords = [];

// Live translation debounce state
// Prevents spamming the translation API every frame — only fires when a
// NEW finalized sign is different from the one already translated.
let lastTranslatedSign = null;
let liveTranslationInFlight = false;

// Video recording state
let mediaRecorder = null;
let recordedChunks = [];
let cameraStream = null;
let isRecording = false;
let isProcessing = false;
let recordTimerInterval = null;
let recordStartTime = 0;

// MediaPipe client-side instances
let mpHands = null;
let mpCamera = null;
let handCanvasCtx = null;

// WebSocket state
let wsSocket = null;
let wsLastSendTime = 0;
const WS_THROTTLE_MS = 100;          // Slightly faster for video-frame pipeline
let wsPrevLandmarks = null;
const WS_MOTION_THRESHOLD = 0.04;
let wsOffCanvas = null;
let wsOffCtx = null;
let wsLastSign = null;
let wsConsecutiveCount = 0;
const WS_CONSECUTIVE_LOCK = 2;

// ── CNN+BiLSTM Frame Capture ──
// Offscreen canvas used to capture and compress video frames as JPEG
// before sending to the backend via WebSocket.
let frameCaptureCanvas = null;
let frameCaptureCtx = null;
const FRAME_CAPTURE_WIDTH = 320;      // Compress to 320×240 for WebSocket transport
const FRAME_CAPTURE_HEIGHT = 240;     // Backend resizes to 224×224 anyway
const FRAME_CAPTURE_QUALITY = 0.7;    // JPEG quality (0.0-1.0)

// Buffer status from server
let wsBufferCount = 0;
let wsBufferRequired = 20;
let wsBufferReady = false;

// Video playback queue instance
var videoQueue = null;

// Current UI mode (1 = Translate Sign, 2 = Generate Sign)
let currentMode = 1;

// Voice recording state
let voiceRecorder = null;
let voiceChunks = [];
let isVoiceRecording = false;
let voiceStream = null;
