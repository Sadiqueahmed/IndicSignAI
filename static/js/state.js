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
const WS_THROTTLE_MS = 150;
let wsPrevLandmarks = null;
const WS_MOTION_THRESHOLD = 0.04;
let wsOffCanvas = null;
let wsOffCtx = null;
let wsLastSign = null;
let wsConsecutiveCount = 0;
const WS_CONSECUTIVE_LOCK = 2;

// Video playback queue instance
var videoQueue = null;

// Current UI mode (1 = Translate Sign, 2 = Generate Sign)
let currentMode = 1;

// Voice recording state
let voiceRecorder = null;
let voiceChunks = [];
let isVoiceRecording = false;
let voiceStream = null;
