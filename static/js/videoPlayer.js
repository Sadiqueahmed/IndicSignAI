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

        this._ready = !!(this.video1 && this.video2);

        if (!this._ready) {
            console.warn('[ISL Player] One or more video elements are missing from the DOM. ' +
                         'Expected: #islVideoPlayer1, #islVideoPlayer2. Player is disabled.');
            return;
        }

        // ── Bind playback events ──────────────────────────────────────────────
        this.video1.addEventListener('ended',  () => this._onEnded());
        this.video2.addEventListener('ended',  () => this._onEnded());
        this.video1.addEventListener('error',  () => this._onError());
        this.video2.addEventListener('error',  () => this._onError());

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

        if (this.idleEl)     this.idleEl.style.display     = 'none';
        if (this.sentenceEl) this.sentenceEl.style.display = 'block';
        if (this.containerEl) this.containerEl.classList.add('playing');

        // Ensure both videos are visible and correctly classed
        this._resetVideoClasses();
        this.video1.style.display = 'block';
        this.video2.style.display = 'block';

        // Load + play first item on video1
        this._load(this.video1, this.queue[0]);
        this._highlightWord(0);

        this.video1.play()
            .then(() => { /* playing */ })
            .catch(e => {
                console.warn(`[ISL Player] play() failed for "${this.queue[0]?.word}":`, e.message);
                this._onError();
            });

        // Preload second item on video2
        this._preload(1);
    }

    stop() {
        if (!this._ready) return;
        try { this.video1.pause(); this.video1.removeAttribute('src'); } catch (_) {}
        try { this.video2.pause(); this.video2.removeAttribute('src'); } catch (_) {}
        this._finish(false);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    _load(videoEl, item) {
        if (!videoEl || !item) return;
        videoEl.src = item.url;
        videoEl.load();
    }

    _preload(index) {
        if (index >= this.queue.length) return;
        const target = this.activeSlot === 1 ? this.video2 : this.video1;
        this._load(target, this.queue[index]);
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

    _onEnded() {
        this.currentIndex++;
        if (this.currentIndex >= this.queue.length) {
            this._finish(true);
            return;
        }

        // Swap active/inactive
        const outgoing = this.activeSlot === 1 ? this.video1 : this.video2;
        const incoming = this.activeSlot === 1 ? this.video2 : this.video1;

        outgoing.classList.replace('active', 'inactive');
        incoming.classList.replace('inactive', 'active');
        this.activeSlot = this.activeSlot === 1 ? 2 : 1;

        this._highlightWord(this.currentIndex);

        incoming.play()
            .then(() => { /* playing */ })
            .catch(e => {
                console.warn(`[ISL Player] play() swap failed for "${this.queue[this.currentIndex]?.word}":`, e.message);
                this._onError();
            });

        this._preload(this.currentIndex + 1);
    }

    _onError() {
        const item = this.queue[this.currentIndex];
        console.warn(
            `[ISL Player] ⚠ No video for word: "${item?.word}" (${item?.url}). Skipping.`
        );
        // Treat as "ended" to advance the queue
        this._onEnded();
    }

    _resetVideoClasses() {
        this.video1.className = 'isl-video active';
        this.video2.className = 'isl-video inactive';
    }

    _finish(showComplete = true) {
        this.isPlaying = false;

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

let videoQueue = null;

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
