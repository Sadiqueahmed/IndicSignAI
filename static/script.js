/**
 * IndicSignAI - Frontend JavaScript
 * Real-time sign language translation application
 */

// Global State
let currentMode = 'sign-to-text';
let isRecording = false;
let recognition = null;
let signHistory = [];
let cameraActive = true;
let statusCheckInterval = null;
let lastDetectedSign = null;
let signConfidence = 0;
let isSpeaking = false;

// DOM Elements
const elements = {
    videoFeed: null,
    cameraPlaceholder: null,
    detectedSignText: null,
    confidenceValue: null,
    confidenceFill: null,
    historyList: null,
    textInput: null,
    translatedText: null,
    currentSignDisplay: null,
    loaderOverlay: null,
    loaderProgressBar: null,
    loaderPercent: null,
    modelStatus: null,
    activeLanguage: null
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initLoader();
    initSpeechRecognition();
    initializeSystem();
    setupEventListeners();
});

// Initialize DOM element references
function initializeElements() {
    elements.videoFeed = document.getElementById('videoFeed');
    elements.cameraPlaceholder = document.getElementById('cameraPlaceholder');
    elements.detectedSignText = document.getElementById('detectedSignText');
    elements.confidenceValue = document.getElementById('confidenceValue');
    elements.confidenceFill = document.getElementById('confidenceFill');
    elements.historyList = document.getElementById('historyList');
    elements.textInput = document.getElementById('textInput');
    elements.translatedText = document.getElementById('translatedText');
    elements.currentSignDisplay = document.getElementById('currentSignDisplay');
    elements.loaderOverlay = document.getElementById('loaderOverlay');
    elements.loaderProgressBar = document.getElementById('loaderProgressBar');
    elements.loaderPercent = document.getElementById('loaderPercent');
    elements.modelStatus = document.getElementById('modelStatus');
    elements.activeLanguage = document.getElementById('activeLanguage');
}

// Setup event listeners
function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === ' ' && currentMode === 'sign-to-text') {
            e.preventDefault();
            speakLastSign();
        }
        if (e.key === 'Escape') {
            stopSpeech();
        }
    });

    // Auto-resize textarea
    if (elements.textInput) {
        elements.textInput.addEventListener('input', autoResizeTextarea);
    }
}

// Loading Animation
function initLoader() {
    let progress = 0;
    const bar = elements.loaderProgressBar;
    const percent = elements.loaderPercent;
    const overlay = elements.loaderOverlay;

    if (!bar || !percent || !overlay) return;

    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            setTimeout(() => {
                overlay.classList.add('hidden');
                // Start status checking after loader
                startStatusChecking();
            }, 500);
        }
        bar.style.width = progress + '%';
        percent.textContent = Math.round(progress) + '%';
    }, 200);
}

// Initialize Backend
function initializeSystem() {
    fetch('/initialize', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            console.log('System initialized:', data);
            if (data.success) {
                updateModelStatus('System Ready');
            } else {
                updateModelStatus('Initialization Failed');
            }
        })
        .catch(err => {
            console.error('Init error:', err);
            updateModelStatus('Connection Error');
        });
}

// Start status checking
function startStatusChecking() {
    checkStatus();
    statusCheckInterval = setInterval(checkStatus, 1000);
}

// Check System Status
function checkStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            updateSystemStatus(data);
            
            // Update sign detection in real-time
            if (currentMode === 'sign-to-text' && data.prediction && data.prediction !== 'Waiting for signs...' && data.prediction !== 'None') {
                const confidence = data.model_confidence || 0;
                if (confidence > 0.3) {
                    updateSignOutput(data.prediction, confidence);
                }
            }
        })
        .catch(err => {
            console.error('Status error:', err);
            updateModelStatus('Connection Lost');
        });
}

// Update system status display
function updateSystemStatus(data) {
    if (elements.modelStatus) {
        let status = 'Initializing...';
        if (data.model && data.initialized) {
            status = 'ISL Model Active';
        } else if (data.initialized) {
            status = 'Camera Ready';
        }
        elements.modelStatus.textContent = status;
    }
}

// Update model status
function updateModelStatus(status) {
    if (elements.modelStatus) {
        elements.modelStatus.textContent = status;
    }
}

// Mode Switching
function switchMode(mode) {
    currentMode = mode;
    
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Update panels
    const signToTextPanel = document.getElementById('panel-sign-to-text');
    const textToSignPanel = document.getElementById('panel-text-to-sign');
    const signToTextOutput = document.getElementById('output-sign-to-text');
    const textToSignOutput = document.getElementById('output-text-to-sign');

    if (signToTextPanel) signToTextPanel.classList.toggle('active', mode === 'sign-to-text');
    if (textToSignPanel) textToSignPanel.classList.toggle('active', mode === 'text-to-sign');
    if (signToTextOutput) signToTextOutput.classList.toggle('active', mode === 'sign-to-text');
    if (textToSignOutput) textToSignOutput.classList.toggle('active', mode === 'text-to-sign');

    // Update status bar
    if (elements.activeLanguage) {
        if (mode === 'sign-to-text') {
            elements.activeLanguage.textContent = 'ISL → English';
        } else {
            const langSelect = document.getElementById('targetLang');
            if (langSelect) {
                const langName = langSelect.options[langSelect.selectedIndex].text;
                elements.activeLanguage.textContent = 'English → ' + langName;
            }
        }
    }
}

// Update Sign Output
function updateSignOutput(sign, confidence) {
    // Only update if sign changed or confidence improved
    if (sign === lastDetectedSign && Math.abs(confidence - signConfidence) < 0.05) {
        return;
    }

    lastDetectedSign = sign;
    signConfidence = confidence;

    const signText = elements.detectedSignText;
    const confValue = elements.confidenceValue;
    const confFill = elements.confidenceFill;

    if (signText) signText.textContent = sign;
    if (confValue) confValue.textContent = Math.round(confidence * 100) + '%';
    if (confFill) confFill.style.width = (confidence * 100) + '%';

    // Add to history if new sign
    if (!signHistory.includes(sign)) {
        signHistory.unshift(sign);
        if (signHistory.length > 10) signHistory.pop();
        updateHistory();
        
        // Auto-speak if enabled (optional feature)
        // speakText(sign);
    }
}

// Update History Display
function updateHistory() {
    const list = elements.historyList;
    if (!list) return;

    if (signHistory.length === 0) {
        list.innerHTML = '<span class="history-empty">No signs detected yet</span>';
        return;
    }

    list.innerHTML = signHistory.map((sign, index) => 
        `<span class="history-item" onclick="speakText('${sign}')" title="Click to speak">${sign}</span>`
    ).join('');
}

// Camera Toggle
function toggleCamera() {
    cameraActive = !cameraActive;
    const feed = elements.videoFeed;
    const placeholder = elements.cameraPlaceholder;
    
    if (!feed || !placeholder) return;

    if (cameraActive) {
        feed.style.display = 'block';
        placeholder.style.display = 'none';
        // Reload video feed
        feed.src = "{{ url_for('video_feed') }}?" + new Date().getTime();
    } else {
        feed.style.display = 'none';
        placeholder.style.display = 'flex';
    }
}

// Text Translation
function translateText() {
    const text = elements.textInput ? elements.textInput.value.trim() : '';
    const targetLang = document.getElementById('targetLang');
    
    if (!text || !targetLang) return;

    const langCode = targetLang.value;

    fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            target_lang: langCode,
            direction: 'en_to_regional'
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success && data.data) {
            if (elements.translatedText) {
                elements.translatedText.textContent = data.data.translated;
            }
            if (elements.currentSignDisplay) {
                elements.currentSignDisplay.textContent = text;
            }
            
            // Animate avatar
            playAvatarAnimation();
        } else {
            console.error('Translation failed:', data.error);
        }
    })
    .catch(err => console.error('Translation error:', err));
}

// Language Change
function changeTargetLanguage() {
    const lang = document.getElementById('targetLang');
    if (!lang || !elements.activeLanguage) return;
    
    const langName = lang.options[lang.selectedIndex].text;
    elements.activeLanguage.textContent = 'English → ' + langName;
}

// Speech Recognition
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.log('Speech recognition not supported');
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        if (elements.textInput) {
            elements.textInput.value = transcript;
        }
        isRecording = false;
        updateRecordingUI();
    };

    recognition.onerror = (e) => {
        console.error('Speech error:', e.error);
        isRecording = false;
        updateRecordingUI();
    };

    recognition.onend = () => {
        isRecording = false;
        updateRecordingUI();
    };
}

function startVoiceInput() {
    if (!recognition) {
        alert('Voice recognition not supported in this browser');
        return;
    }
    
    try {
        if (isRecording) {
            recognition.stop();
        } else {
            recognition.start();
            isRecording = true;
        }
        updateRecordingUI();
    } catch (e) {
        console.error('Voice input error:', e);
    }
}

function updateRecordingUI() {
    const btn = document.querySelector('.action-btn[onclick="startVoiceInput()"]');
    if (btn) {
        btn.classList.toggle('recording', isRecording);
        btn.innerHTML = isRecording ? '<i class="fa-solid fa-stop"></i>' : '<i class="fa-solid fa-microphone"></i>';
        btn.title = isRecording ? 'Stop Recording' : 'Voice Input';
    }
}

// Text-to-Speech
function speakOutput() {
    const text = elements.detectedSignText ? elements.detectedSignText.textContent : '';
    if (!text || text === 'Waiting for signs...') return;
    
    speakText(text);
}

function speakLastSign() {
    if (lastDetectedSign) {
        speakText(lastDetectedSign);
    }
}

function speakText(text) {
    if (!text || isSpeaking) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    const lang = document.getElementById('targetLang');
    const langCode = lang ? lang.value : 'en';
    
    const langMap = {
        'assamese': 'as-IN',
        'hindi': 'hi-IN',
        'manipuri': 'bn-IN',
        'nepali': 'ne-NP',
        'marathi': 'mr-IN',
        'odia': 'or-IN',
        'mizorami': 'en-US',
        'gujarati': 'gu-IN',
        'tamil': 'ta-IN',
        'telugu': 'te-IN',
        'bengali': 'bn-IN',
        'meitei_lon': 'en-US',
        'dzongkha': 'en-US',
        'en': 'en-US'
    };
    
    utterance.lang = langMap[langCode] || 'en-US';
    utterance.rate = 0.9;
    utterance.pitch = 1;
    
    isSpeaking = true;
    
    utterance.onend = () => {
        isSpeaking = false;
    };
    
    utterance.onerror = () => {
        isSpeaking = false;
    };
    
    window.speechSynthesis.speak(utterance);
}

function stopSpeech() {
    window.speechSynthesis.cancel();
    isSpeaking = false;
}

// Utility Functions
function clearText() {
    if (elements.textInput) {
        elements.textInput.value = '';
        autoResizeTextarea();
    }
}

function copyOutput() {
    const text = elements.detectedSignText ? elements.detectedSignText.textContent : '';
    if (!text) return;
    
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!');
    }).catch(err => {
        console.error('Copy failed:', err);
    });
}

function autoResizeTextarea() {
    const textarea = elements.textInput;
    if (!textarea) return;
    
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Settings Panel
function toggleSettings() {
    // Create settings modal if it doesn't exist
    let modal = document.getElementById('settingsModal');
    if (!modal) {
        modal = createSettingsModal();
        document.body.appendChild(modal);
    }
    modal.classList.toggle('active');
}

function createSettingsModal() {
    const modal = document.createElement('div');
    modal.id = 'settingsModal';
    modal.className = 'settings-modal';
    modal.innerHTML = `
        <div class="settings-content">
            <div class="settings-header">
                <h3>Settings</h3>
                <button onclick="toggleSettings()" class="close-btn">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
            <div class="settings-body">
                <div class="setting-item">
                    <label>Auto-speak detected signs</label>
                    <input type="checkbox" id="autoSpeak" onchange="toggleAutoSpeak()">
                </div>
                <div class="setting-item">
                    <label>Speech rate</label>
                    <input type="range" id="speechRate" min="0.5" max="1.5" step="0.1" value="0.9">
                </div>
                <div class="setting-item">
                    <label>Show confidence scores</label>
                    <input type="checkbox" id="showConfidence" checked>
                </div>
            </div>
        </div>
    `;
    return modal;
}

function toggleAutoSpeak() {
    const checkbox = document.getElementById('autoSpeak');
    if (checkbox) {
        localStorage.setItem('autoSpeak', checkbox.checked);
    }
}

// Avatar Animation Controls
function playAnimation() {
    const arm = document.querySelector('.right-arm');
    if (arm) {
        arm.style.animationPlayState = 'running';
    }
}

function pauseAnimation() {
    const arm = document.querySelector('.right-arm');
    if (arm) {
        arm.style.animationPlayState = 'paused';
    }
}

function replayAnimation() {
    const arm = document.querySelector('.right-arm');
    if (arm) {
        arm.style.animation = 'none';
        setTimeout(() => {
            arm.style.animation = 'armGesture 2s ease-in-out infinite';
        }, 10);
    }
}

function playAvatarAnimation() {
    replayAnimation();
}

// Notification system
function showNotification(message) {
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.className = 'notification';
        document.body.appendChild(notification);
    }
    
    notification.textContent = message;
    notification.classList.add('show');
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Keyboard shortcuts help
function showShortcuts() {
    const shortcuts = `
Keyboard Shortcuts:
Space - Speak last detected sign
Esc - Stop speech
Ctrl + C - Copy output
Ctrl + Enter - Translate (in text mode)
    `;
    alert(shortcuts);
}

// Export functions for global access
window.switchMode = switchMode;
window.toggleCamera = toggleCamera;
window.translateText = translateText;
window.changeTargetLanguage = changeTargetLanguage;
window.startVoiceInput = startVoiceInput;
window.clearText = clearText;
window.copyOutput = copyOutput;
window.speakOutput = speakOutput;
window.toggleSettings = toggleSettings;
window.playAnimation = playAnimation;
window.pauseAnimation = pauseAnimation;
window.replayAnimation = replayAnimation;
window.speakText = speakText;
