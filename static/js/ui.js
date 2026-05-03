/**
 * ui.js — All UI Logic: Mode Switching, Recording, Sentence, Translation, Voice, Toast
 * Depends on: state.js, camera.js, websocket.js, videoPlayer.js
 */

// ========== MODE SWITCHING ==========
function switchMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;
    document.getElementById('mode1Panel').classList.toggle('active', mode === 1);
    document.getElementById('mode2Panel').classList.toggle('active', mode === 2);
    document.getElementById('modeBtn1').classList.toggle('active', mode === 1);
    document.getElementById('modeBtn2').classList.toggle('active', mode === 2);

    if (mode === 1) {
        startCamera();
        if (isVoiceRecording) stopVoiceRecording();
        showToast('Switched to Translate Sign mode', 'success');
    } else {
        stopCamera();
        showToast('Switched to Generate Sign mode', 'success');
    }
}

// ========== INIT ==========
document.addEventListener('DOMContentLoaded', function () {
    videoQueue = new VideoQueueManager();
    initializeSystem();
    initWebSocket();
    startCamera();
    syncSentence();
});

async function initializeSystem() {
    try {
        const response = await fetch('/initialize', { method: 'POST' });
        const data = await response.json();
        if (data.success) showToast('System ready!', 'success');
    } catch (error) {
        console.error('Initialization error:', error);
    }
}

// ========== RECORD / STOP ==========
function toggleRecording() { isRecording ? stopRecording() : startRecording(); }

function startRecording() {
    if (!cameraStream) { showToast('Camera not available', 'error'); return; }
    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9'
        : MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '';
    mediaRecorder = new MediaRecorder(cameraStream, mimeType ? { mimeType } : {});
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => handleRecordingStopped();
    mediaRecorder.start(200);
    isRecording = true;
    document.getElementById('recordBtn').classList.add('recording');
    document.getElementById('cameraContainer').classList.add('recording');
    const badge = document.getElementById('detectionBadge');
    badge.classList.remove('active');
    badge.innerHTML = '<i class="fa-solid fa-circle" style="color:#ef4444"></i><span>Recording...</span>';
    recordStartTime = Date.now();
    const timerEl = document.getElementById('recordTimer');
    timerEl.classList.add('visible'); timerEl.textContent = '0:00';
    recordTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        timerEl.textContent = Math.floor(elapsed / 60) + ':' + String(elapsed % 60).padStart(2, '0');
    }, 500);
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
    document.getElementById('recordBtn').classList.remove('recording');
    document.getElementById('cameraContainer').classList.remove('recording');
    clearInterval(recordTimerInterval);
    document.getElementById('recordTimer').classList.remove('visible');
}

async function handleRecordingStopped() {
    if (recordedChunks.length === 0) { showToast('No video data recorded', 'error'); resetBadge(); return; }
    const blob = new Blob(recordedChunks, { type: recordedChunks[0].type || 'video/webm' });
    recordedChunks = [];
    isProcessing = true;
    document.getElementById('processingOverlay').classList.add('visible');

    const formData = new FormData();
    formData.append('video', blob, 'sign_recording.webm');
    formData.append('target_lang', document.getElementById('targetLanguage').value);

    try {
        const response = await fetch('/api/process-video', { method: 'POST', body: formData });
        const data = await response.json();
        document.getElementById('processingOverlay').classList.remove('visible');
        isProcessing = false;

        if (data.status === 'error') {
            document.getElementById('currentSign').textContent = '--';
            document.getElementById('signsDetail').textContent = data.message || 'Processing failed';
            showToast(data.message || 'No signs detected', 'error');
        } else if (data.success && data.signs && data.signs.length > 0) {
            currentSign = data.signs.join(' ');
            document.getElementById('currentSign').textContent = data.signs.join(', ');
            document.getElementById('signsDetail').textContent =
                data.signs.length + ' sign' + (data.signs.length > 1 ? 's' : '') + ' detected';
            document.getElementById('addWordBtn').disabled = false;
            const textarea = document.getElementById('sentenceText');
            if (textarea && data.corrected) textarea.value = data.corrected;
            sentenceWords = data.signs;
            const wc = document.getElementById('wordCount');
            if (wc) wc.textContent = data.signs.length + ' word' + (data.signs.length > 1 ? 's' : '');
            if (data.corrected) document.getElementById('originalText').textContent = data.corrected;
            if (data.translated) document.getElementById('translatedText').textContent = data.translated;
            if (data.corrected && videoQueue) {
                const ok = videoQueue.enqueueSentence(data.corrected);
                if (ok) videoQueue.play();
            }
            showToast('Detected: ' + data.signs.join(', '), 'success');
        } else {
            document.getElementById('currentSign').textContent = '--';
            document.getElementById('signsDetail').textContent = 'No signs detected  -  try recording longer';
            showToast('No signs detected.', 'error');
        }
    } catch (err) {
        document.getElementById('processingOverlay').classList.remove('visible');
        isProcessing = false;
        console.error('Process video error:', err);
        showToast('Processing failed', 'error');
    }
    resetBadge();
}

function resetBadge() {
    const badge = document.getElementById('detectionBadge');
    badge.classList.add('active');
    badge.innerHTML = '<i class="fa-solid fa-video"></i><span>Camera ready  -  tap record</span>';
}

// ========== SENTENCE MANAGEMENT ==========
async function addWordToSentence(word) {
    try {
        const resp = await fetch('/api/sentence', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'add', sign: word })
        });
        const data = await resp.json();
        if (data.success) updateSentenceUI(data.words, data.sentence);
    } catch (e) { console.error('Add word error:', e); }
}

function addCurrentWord() {
    if (currentSign) { addWordToSentence(currentSign); showToast('Added: ' + currentSign, 'success'); }
}

async function undoLastWord() {
    try {
        const resp = await fetch('/api/sentence', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'undo' })
        });
        const data = await resp.json();
        if (data.success) { updateSentenceUI(data.words, data.sentence); showToast('Undone', 'success'); }
    } catch (e) { console.error('Undo error:', e); }
}

async function clearSentence() {
    try {
        const resp = await fetch('/api/sentence', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'clear' })
        });
        const data = await resp.json();
        if (data.success) { updateSentenceUI([], ''); showToast('Cleared', 'success'); }
    } catch (e) { console.error('Clear error:', e); }
}

function copySentence() {
    const text = document.getElementById('sentenceText');
    if (text && text.value) navigator.clipboard.writeText(text.value).then(() => showToast('Copied!', 'success'));
    else showToast('Nothing to copy', 'error');
}

function speakSentence() {
    const text = document.getElementById('sentenceText');
    if (text && text.value) {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text.value);
        u.lang = 'en-US'; u.rate = 0.85;
        speechSynthesis.speak(u);
        showToast('Speaking sentence...', 'success');
    } else showToast('No sentence to speak', 'error');
}

async function syncSentence() {
    try {
        const resp = await fetch('/api/sentence');
        const data = await resp.json();
        if (data.success) updateSentenceUI(data.words, data.sentence);
    } catch (e) { console.error('Sync error:', e); }
}

function updateSentenceUI(words, sentence) {
    sentenceWords = words || [];
    const ta = document.getElementById('sentenceText');
    const wc = document.getElementById('wordCount');
    if (ta) ta.value = sentence || '';
    if (wc) wc.textContent = sentenceWords.length + ' word' + (sentenceWords.length !== 1 ? 's' : '');
}

// ========== TRANSLATION ==========
async function translateCurrent() {
    if (!currentSign) return;
    const targetLang = document.getElementById('targetLanguage').value;
    try {
        const resp = await fetch('/api/translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentSign, target_lang: targetLang, direction: 'en_to_regional' })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[translateCurrent] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }
        const data = await resp.json();
        if (data.success) {
            document.getElementById('translatedText').textContent = data.data.translated;
        } else {
            console.error('[translateCurrent] Backend error:', data.error);
            _showTranslationError();
        }
    } catch (error) {
        console.error('[translateCurrent] Network error:', error);
        _showTranslationError();
    }
}

async function translateSentence() {
    const sentence = document.getElementById('sentenceText').value.trim();
    if (!sentence) { showToast('Build a sentence first', 'error'); return; }
    const targetLang = document.getElementById('targetLanguage').value;
    try {
        showToast('Translating...', 'success');
        const resp = await fetch('/api/translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: sentence, target_lang: targetLang, direction: 'en_to_regional' })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[translateSentence] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }
        const data = await resp.json();
        if (data.success) {
            document.getElementById('originalText').textContent = sentence;
            document.getElementById('translatedText').textContent = data.data.translated;
            showToast('Translated!', 'success');
        } else {
            console.error('[translateSentence] Backend error:', data.error);
            _showTranslationError();
        }
    } catch (error) {
        console.error('[translateSentence] Network error:', error);
        _showTranslationError();
    }
}

// ========== SPEECH & SIGN PLAYBACK ==========
function speakDetectedSign() {
    if (!currentSign) { showToast('No sign detected yet', 'error'); return; }
    const u = new SpeechSynthesisUtterance(currentSign);
    u.lang = 'en-US'; u.rate = 0.9;
    speechSynthesis.speak(u);
}

function _isNonEnglish(text) {
    return /[^\x00-\x7F]/.test(text);
}

async function convertTextToSign() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) { showToast('Please enter text', 'error'); return; }

    const voiceLang = document.getElementById('voiceLanguage').value;
    const sourceLang = _isNonEnglish(text) ? voiceLang : 'english';
    const bar = document.getElementById('voiceProcessingBar');
    const barText = document.getElementById('voiceProcessingText');

    // ── English shortcut: no API round-trip needed ────────────────────────
    if (sourceLang === 'english') {
        showToast('Playing ISL video...', 'success');
        playISLSequence(text);
        return;
    }

    // ── Non-English: translate first, then play ───────────────────────────
    bar.classList.add('visible');
    barText.textContent = `Translating ${sourceLang} → English → ISL...`;

    let corrected = text;
    try {
        const resp = await fetch('/api/process-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source_lang: sourceLang })
        });
        const data = await resp.json();
        if (data.success) {
            corrected = data.corrected || data.english_text || text;
            if (data.corrected)
                document.getElementById('m2OriginalText').textContent = data.corrected;
            if (data.input_text)
                document.getElementById('m2TranslatedText').textContent = data.input_text;
            showToast(`"${text}" → "${corrected}" → Playing ISL`, 'success');
        } else {
            showToast(data.error || 'Translation failed, playing original', 'error');
        }
    } catch (err) {
        console.error('[convertTextToSign] Translation API error:', err);
        showToast('Translation failed — playing original text', 'error');
    } finally {
        bar.classList.remove('visible');
    }

    // Always attempt playback, even if translation failed
    playISLSequence(corrected);
}

async function showSign(signName) {
    currentSign = signName;
    document.getElementById('originalText').textContent = signName;
    translateCurrent();
    try {
        playISLSequence(signName);
    } catch (e) { console.error('showSign error:', e); }
}

// ========== MODE 2 TOOLBAR ==========
function m2AddWord() {
    const el = document.getElementById('textInput');
    el.focus();
    el.value += (el.value.endsWith(' ') || el.value === '' ? '' : ' ');
    showToast('Ready to type', 'success');
}
function m2Undo() {
    const el = document.getElementById('textInput');
    let words = el.value.trim().split(/\s+/);
    if (words.length > 0 && words[0] !== '') {
        words.pop();
        el.value = words.join(' ') + (words.length > 0 ? ' ' : '');
        showToast('Undone last word', 'success');
    }
}
function m2Clear() {
    document.getElementById('textInput').value = '';
    document.getElementById('m2OriginalText').textContent = '--';
    document.getElementById('m2TranslatedText').textContent = '--';
    showToast('Cleared text', 'success');
}
function m2Copy() {
    const val = document.getElementById('textInput').value;
    if (val) navigator.clipboard.writeText(val).then(() => showToast('Copied to clipboard!', 'success'));
    else showToast('Nothing to copy', 'error');
}
function m2Speak() {
    const val = document.getElementById('textInput').value;
    if (val) {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(val);
        u.lang = 'en-US';
        speechSynthesis.speak(u);
        showToast('Speaking text...', 'success');
    } else showToast('No text to speak', 'error');
}
async function m2Translate() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) { showToast('Enter text first', 'error'); return; }
    const tgt = document.getElementById('voiceLanguage').value;
    try {
        showToast('Translating...', 'success');
        const resp = await fetch('/api/translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target_lang: tgt === 'english' ? 'hindi' : tgt, direction: 'en_to_regional' })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[m2Translate] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }
        const data = await resp.json();
        if (data.success) {
            document.getElementById('m2TranslatedText').textContent = data.data.translated;
            showToast('Translated!', 'success');
        } else {
            console.error('[m2Translate] Backend error:', data.error);
            _showTranslationError();
        }
    } catch (error) {
        console.error('[m2Translate] Network error:', error);
        _showTranslationError();
    }
}

/** Shared helper: show a red toast AND write fallback text to any visible translation output. */
function _showTranslationError() {
    showToast('Translation Service Unavailable', 'error');
    // Write a visible indicator into the output fields if they exist
    const fields = ['translatedText', 'm2TranslatedText'];
    fields.forEach(id => {
        const el = document.getElementById(id);
        if (el && (el.textContent === '--' || el.textContent === '' || el.textContent === 'Translating...')) {
            el.style.color = 'var(--color-error, #ef4444)';
            el.textContent = 'Translation Service Unavailable';
            setTimeout(() => { el.style.color = ''; if (el.textContent === 'Translation Service Unavailable') el.textContent = '--'; }, 4000);
        }
    });
}
async function m2FixGrammar() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) { showToast('Enter text first', 'error'); return; }
    try {
        showToast('Fixing grammar...', 'success');
        const resp = await fetch('/api/correct-and-translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words: text.split(/\s+/), target_lang: 'english' })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[m2FixGrammar] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }
        const data = await resp.json();
        if (data.success) {
            document.getElementById('textInput').value = data.data.corrected;
            document.getElementById('m2OriginalText').textContent = data.data.corrected;
            showToast('Grammar fixed!', 'success');
        } else {
            console.error('[m2FixGrammar] Backend error:', data.error);
            showToast(data.error || 'Grammar check failed', 'error');
        }
    } catch (e) {
        console.error('[m2FixGrammar] Network error:', e);
        showToast('Grammar check failed', 'error');
    }
}

// ========== PUSH-TO-TALK VOICE RECORDING ==========
async function toggleVoiceRecording() {
    if (isVoiceRecording) stopVoiceRecording(); else await startVoiceRecording();
}

async function startVoiceRecording() {
    try {
        voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) { showToast('Microphone access denied', 'error'); return; }
    voiceChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
    voiceRecorder = new MediaRecorder(voiceStream, mimeType ? { mimeType } : {});
    voiceRecorder.ondataavailable = (e) => { if (e.data.size > 0) voiceChunks.push(e.data); };
    voiceRecorder.onstop = () => handleVoiceStopped();
    voiceRecorder.start(200);
    isVoiceRecording = true;
    const btn = document.getElementById('micBtn');
    btn.classList.add('recording');
    document.getElementById('micIcon').className = 'fa-solid fa-stop';
    showToast('🎤 Listening... tap again to stop', 'success');
}

function stopVoiceRecording() {
    if (voiceRecorder && voiceRecorder.state !== 'inactive') voiceRecorder.stop();
    if (voiceStream) { voiceStream.getTracks().forEach(t => t.stop()); voiceStream = null; }
    isVoiceRecording = false;
    document.getElementById('micBtn').classList.remove('recording');
    document.getElementById('micIcon').className = 'fa-solid fa-microphone';
}

function encodeWAV(samples) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (v, offset, str) => { for (let i = 0; i < str.length; i++) v.setUint8(offset + i, str.charCodeAt(i)); };
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
}

async function handleVoiceStopped() {
    if (voiceChunks.length === 0) { showToast('No audio recorded', 'error'); return; }
    const webmBlob = new Blob(voiceChunks, { type: voiceChunks[0].type || 'audio/webm' });
    voiceChunks = [];
    const bar = document.getElementById('voiceProcessingBar');
    const barText = document.getElementById('voiceProcessingText');
    bar.classList.add('visible');
    barText.textContent = 'Listening & Transcribing...';
    let wavBlob;
    try {
        const arrayBuffer = await webmBlob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        wavBlob = encodeWAV(audioBuffer.getChannelData(0));
    } catch (err) {
        console.error('Audio decoding failed frontend:', err);
        wavBlob = webmBlob;
    }
    const sourceLang = document.getElementById('voiceLanguage').value;
    const formData = new FormData();
    formData.append('audio', wavBlob, 'voice_input.wav');
    formData.append('source_lang', sourceLang);
    try {
        const resp = await fetch('/api/speech-to-text', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!data.success) {
            bar.classList.remove('visible');
            showToast(data.error || 'Speech-to-text failed', 'error');
            return;
        }
        showToast(`Heard: "${data.text}"`, 'success');
        document.getElementById('textInput').value = data.text;
        await convertTextToSign();
    } catch (err) {
        bar.classList.remove('visible');
        console.error('STT error:', err);
        showToast('Speech processing failed', 'error');
    }
}

// ========== NLP GRAMMAR CORRECTION ==========
async function fixGrammar() {
    if (!sentenceWords || sentenceWords.length === 0) { showToast('Build a sentence first', 'error'); return; }
    try {
        showToast('Fixing grammar...', 'success');
        const targetLang = document.getElementById('targetLanguage').value;
        const resp = await fetch('/api/correct-and-translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words: sentenceWords, target_lang: targetLang })
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            console.error('[fixGrammar] HTTP', resp.status, errData);
            _showTranslationError();
            return;
        }
        const data = await resp.json();
        if (data.success && data.data) {
            document.getElementById('sentenceText').value = data.data.corrected;
            document.getElementById('originalText').textContent = data.data.corrected;
            if (data.data.translated) document.getElementById('translatedText').textContent = data.data.translated;
            showToast('Grammar corrected & translated!', 'success');
            const builder = document.getElementById('sentenceBuilder');
            if (builder) { builder.classList.add('highlight'); setTimeout(() => builder.classList.remove('highlight'), 800); }
        } else {
            console.error('[fixGrammar] Backend error:', data.error);
            showToast(data.error || 'Correction failed', 'error');
        }
    } catch (e) {
        console.error('[fixGrammar] Network error:', e);
        showToast('Grammar fix failed', 'error');
    }
}

async function animateSentence() {
    const sentence = document.getElementById('sentenceText').value.trim();
    if (!sentence) { showToast('Build a sentence first', 'error'); return; }
    showToast('Preparing ISL animation...', 'success');
    playISLSequence(sentence);
}

// ========== TOAST ==========
function showToast(message, type) {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    toast.className = 'toast ' + (type || 'success');
    toastMessage.textContent = message;
    const icon = toast.querySelector('i');
    if (type === 'success') icon.className = 'fa-solid fa-circle-check';
    else if (type === 'error') icon.className = 'fa-solid fa-circle-exclamation';
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// ========== KEYBOARD & EVENT LISTENERS ==========
document.getElementById('textInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') convertTextToSign();
});
document.getElementById('targetLanguage').addEventListener('change', function () {
    // Reset live translation debounce so the current sign is re-translated
    // in the newly selected language
    lastTranslatedSign = null;
    if (currentSign && currentMode === 1) {
        _autoTranslateLiveSign(currentSign);
    }
    // Also re-translate any sentence in the builder
    const sentence = document.getElementById('sentenceText').value.trim();
    if (sentence) translateSentence();
});
