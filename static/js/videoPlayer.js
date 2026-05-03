/**
 * videoPlayer.js — ISL Video Playback Queue (Dual-Buffer Ping-Pong)
 * Depends on: ui.js (showToast) — must be loaded after the DOM is ready.
 *
 * Initialised lazily on first use so that constructor errors can never
 * silently break other parts of the application.
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// Text Normalizer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Lowercases, strips punctuation, collapses whitespace, and splits a sentence
 * into an array of individual words ready for video lookup.
 *
 * @param {string} sentence
 * @returns {string[]}
 */
function normalizeText(sentence) {
    if (!sentence || typeof sentence !== 'string') return [];
    return sentence
        .toLowerCase()
        .replace(/[.,/#!$%^&*;:{}=\-_`~()?'"]/g, '')
        .replace(/\s{2,}/g, ' ')
        .trim()
        .split(' ')
        .filter(w => w.length > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoQueueManager
// ─────────────────────────────────────────────────────────────────────────────

class VideoQueueManager {
    constructor() {
        // ── Grab DOM elements (may be null if called before DOMContentLoaded) ──
        this.video1       = document.getElementById('islVideoPlayer1');
        this.video2       = document.getElementById('islVideoPlayer2');
        this.sentenceEl   = document.getElementById('videoSentenceOverlay');
        this.progressEl   = document.getElementById('videoQueueProgress');
        this.idleEl       = document.getElementById('videoIdle');
        this.containerEl  = document.getElementById('videoPlayerContainer');

        this.queue        = [];
        this.currentIndex = 0;
        this.isPlaying    = false;
        this.activeSlot   = 1;   // which video element is "on screen"
        this._skipGuard   = false; // prevents infinite _onError → _onEnded loops
        this._advancing   = false; // prevents double-advance from stale events

        this._ready = !!(this.video1 && this.video2);

        if (!this._ready) {
            console.warn('[ISL Player] One or more video elements are missing from the DOM. ' +
                         'Expected: #islVideoPlayer1, #islVideoPlayer2. Player is disabled.');
            return;
        }

        // NOTE: We do NOT bind global 'ended'/'error' events here.
        // They are attached per-load in _loadAndPlay() to avoid stale event races.

        console.log('[ISL Player] VideoQueueManager initialised ✔');
    }

    // ── Public: load and play a sentence ─────────────────────────────────────

    enqueueSentence(sentence) {
        if (!this._ready) return false;

        const words = normalizeText(sentence);
        if (words.length === 0) return false;

        this.queue = [];

        // Build sentence overlay spans
        if (this.sentenceEl) {
            this.sentenceEl.innerHTML = '';
            words.forEach(word => {
                const span = document.createElement('span');
                span.className   = 'word-span';
                span.textContent = word;
                this.sentenceEl.appendChild(span);

                // Title-case the filename for lookup (Hello.mp4, Good.mp4 …)
                const filename = word.charAt(0).toUpperCase() + word.slice(1);
                this.queue.push({ word, url: `/api/video/${filename}.mp4`, spanEl: span });
            });
        } else {
            // No overlay element — build queue without spans
            words.forEach(word => {
                const filename = word.charAt(0).toUpperCase() + word.slice(1);
                this.queue.push({ word, url: `/api/video/${filename}.mp4`, spanEl: null });
            });
        }

        this.currentIndex = 0;
        return true;
    }

    play() {
        if (!this._ready || this.queue.length === 0) return;

        this.isPlaying  = true;
        this.activeSlot = 1;
        this._skipGuard = false;
        this._advancing = false;

        if (this.idleEl)     this.idleEl.style.display     = 'none';
        if (this.sentenceEl) this.sentenceEl.style.display = 'block';
        if (this.containerEl) this.containerEl.classList.add('playing');

        // Ensure both videos are visible and correctly classed
        this._resetVideoClasses();
        this.video1.style.display = 'block';
        this.video2.style.display = 'block';

        // Load + play first item on video1, waiting for it to be ready
        this._highlightWord(0);
        this._loadAndPlay(this.video1, this.queue[0]);
    }

    stop() {
        if (!this._ready) return;
        this._cleanupVideo(this.video1);
        this._cleanupVideo(this.video2);
        this._finish(false);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Remove all per-load handlers and clear the video source. */
    _cleanupVideo(videoEl) {
        if (!videoEl) return;
        try { videoEl.pause(); } catch (_) {}
        videoEl._endedHandler && videoEl.removeEventListener('ended', videoEl._endedHandler);
        videoEl._errorHandler && videoEl.removeEventListener('error', videoEl._errorHandler);
        videoEl._readyHandler && videoEl.removeEventListener('canplaythrough', videoEl._readyHandler);
        if (videoEl._loadTimeout) { clearTimeout(videoEl._loadTimeout); videoEl._loadTimeout = null; }
        videoEl._endedHandler = null;
        videoEl._errorHandler = null;
        videoEl._readyHandler = null;
        try { videoEl.removeAttribute('src'); videoEl.load(); } catch (_) {}
    }

    _loadAndPlay(videoEl, item) {
        if (!videoEl || !item) return;

        // Clean up any previous load on this element
        this._cleanupVideo(videoEl);
        this._advancing = false;

        // ── Error handler (video file missing → fingerspelling fallback) ──
        videoEl._errorHandler = () => {
            console.warn(`[ISL Player] ⚠ No video for word: "${item.word}" (${item.url}).`);

            // ── FINGERSPELLING FALLBACK ──────────────────────────────────
            if (!item._isLetter && item.word && item.word.length > 0) {
                const letters = item.word.toUpperCase().split('').filter(c => /[A-Z]/.test(c));
                if (letters.length > 0) {
                    console.log(`[ISL Player] 🔤 Fingerspelling "${item.word}" → [${letters.join(', ')}]`);

                    const letterItems = letters.map(letter => ({
                        word: letter.toLowerCase(),
                        url: `/api/video/${letter}.mp4`,
                        spanEl: item.spanEl,
                        _isLetter: true
                    }));

                    this.queue.splice(this.currentIndex, 1, ...letterItems);
                    this._highlightWord(this.currentIndex);

                    // Play the first letter on the SAME video element
                    setTimeout(() => {
                        this._loadAndPlay(videoEl, this.queue[this.currentIndex]);
                    }, 50);
                    return;
                }
            }

            // No fingerspelling possible — skip
            this._safeAdvance();
        };
        videoEl.addEventListener('error', videoEl._errorHandler, { once: true });

        // ── Ended handler (video finished playing → advance to next) ──
        videoEl._endedHandler = () => {
            this._safeAdvance();
        };
        videoEl.addEventListener('ended', videoEl._endedHandler, { once: true });

        // Set source and start loading
        videoEl.src = item.url;
        videoEl.load();

        // ── Ready handler (video buffered enough to play) ──
        videoEl._readyHandler = () => {
            videoEl.removeEventListener('canplaythrough', videoEl._readyHandler);
            videoEl._readyHandler = null;
            if (videoEl._loadTimeout) { clearTimeout(videoEl._loadTimeout); videoEl._loadTimeout = null; }
            videoEl.play()
                .then(() => { /* playing */ })
                .catch(e => {
                    console.warn(`[ISL Player] play() failed for "${item.word}":`, e.message);
                    this._safeAdvance();
                });
        };
        videoEl.addEventListener('canplaythrough', videoEl._readyHandler);

        // Safety timeout — skip if video doesn't load in 3 seconds
        videoEl._loadTimeout = setTimeout(() => {
            videoEl.removeEventListener('canplaythrough', videoEl._readyHandler);
            videoEl._readyHandler = null;
            videoEl._loadTimeout = null;
            console.warn(`[ISL Player] Timeout loading "${item.word}". Skipping.`);
            this._safeAdvance();
        }, 3000);
    }

    /** Advance to next item, with guard against double-calls from stale events. */
    _safeAdvance() {
        if (this._advancing) return; // prevent double-advance
        this._advancing = true;

        this.currentIndex++;
        if (this.currentIndex >= this.queue.length) {
            this._finish(true);
            return;
        }

        // Swap active/inactive video elements (ping-pong)
        const incoming = this.activeSlot === 1 ? this.video2 : this.video1;
        const outgoing = this.activeSlot === 1 ? this.video1 : this.video2;

        outgoing.classList.replace('active', 'inactive');
        incoming.classList.replace('inactive', 'active');
        this.activeSlot = this.activeSlot === 1 ? 2 : 1;

        this._highlightWord(this.currentIndex);
        this._loadAndPlay(incoming, this.queue[this.currentIndex]);
    }

    _highlightWord(index) {
        if (!this.sentenceEl) return;
        this.queue.forEach(item => item.spanEl?.classList.remove('active'));
        this.queue[index]?.spanEl?.classList.add('active');

        if (this.progressEl) {
            this.progressEl.textContent = `${index + 1} / ${this.queue.length}`;
            this.progressEl.style.display = 'block';
        }
    }

    _resetVideoClasses() {
        this.video1.className = 'isl-video active';
        this.video2.className = 'isl-video inactive';
    }

    _finish(showComplete = true) {
        this.isPlaying = false;
        this._advancing = false;

        this._cleanupVideo(this.video1);
        this._cleanupVideo(this.video2);

        if (this.video1) this.video1.style.display = 'none';
        if (this.video2) this.video2.style.display = 'none';
        if (this.sentenceEl) this.sentenceEl.style.display = 'none';
        if (this.progressEl) this.progressEl.style.display = 'none';
        if (this.idleEl)     this.idleEl.style.display     = 'flex';
        if (this.containerEl) this.containerEl.classList.remove('playing');

        this.queue.forEach(item => item.spanEl?.classList.remove('active'));

        if (showComplete && typeof showToast === 'function') {
            showToast('Animation complete!', 'success');
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global instance — initialised AFTER DOM is fully parsed
// (scripts are at bottom of <body>, so document.readyState is 'complete')
// ─────────────────────────────────────────────────────────────────────────────

var videoQueue = null;

function _initVideoQueue() {
    try {
        videoQueue = new VideoQueueManager();
    } catch (err) {
        console.error('[ISL Player] Failed to initialise VideoQueueManager:', err);
        videoQueue = null;
    }
}

// Safe to call immediately since scripts are at the bottom of <body>
_initVideoQueue();

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Play an ISL animation sequence for the given sentence.
 * Safe to call from any context — will log a warning if the player
 * is unavailable instead of throwing.
 *
 * @param {string} sentence - Raw input sentence (any case / punctuation).
 */
function playISLSequence(sentence) {
    if (!videoQueue || !videoQueue._ready) {
        console.warn('[ISL Player] Player not ready. Attempting re-init…');
        _initVideoQueue();
        if (!videoQueue || !videoQueue._ready) {
            if (typeof showToast === 'function')
                showToast('Video player not available', 'error');
            return;
        }
    }

    if (videoQueue.isPlaying) {
        videoQueue.stop();
    }

    if (videoQueue.enqueueSentence(sentence)) {
        videoQueue.play();
    } else {
        if (typeof showToast === 'function')
            showToast('No words to animate', 'error');
    }
}
