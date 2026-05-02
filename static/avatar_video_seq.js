/**
 * IndicSignAI - ISL Video Avatar Sequencer
 * Plays authentic Indian Sign Language MP4 animations sequentially
 */

class ISLVideoAvatar {
    constructor(containerId, videoId, placeholderId) {
        this.container = document.getElementById(containerId);
        this.videoPlayer = document.getElementById(videoId);
        this.placeholder = document.getElementById(placeholderId);

        this.videoQueue = [];
        this.isPlaying = false;
        this.availableVideos = new Set();
        this.isInitialized = false;

        this.init();
    }

    async init() {
        if (!this.videoPlayer || !this.placeholder) {
            console.error('ISL Video Avatar elements not found');
            return;
        }

        // Listen for video ending to play next in queue
        this.videoPlayer.addEventListener('ended', () => {
            this.playNextVideo();
        });

        // Hide video initially
        this.videoPlayer.style.display = 'none';
        this.placeholder.style.display = 'flex';

        try {
            await this.fetchAvailableVideos();
            this.isInitialized = true;
            console.log(`ISL Video Avatar initialized with ${this.availableVideos.size} signs`);
        } catch (error) {
            console.error('Failed to initialize ISL Video Avatar:', error);
            this.showFallback('Failed to load video dictionary');
        }
    }

    async fetchAvailableVideos() {
        try {
            const response = await fetch('/api/available-videos');
            const data = await response.json();

            if (data.success && data.videos) {
                // Store standard uppercase versions for easier matching
                data.videos.forEach(v => {
                    this.availableVideos.add(v.toUpperCase());
                });
            } else {
                throw new Error(data.error || 'Failed to fetch video list');
            }
        } catch (error) {
            console.warn('Could not fetch video list:', error);
            throw error;
        }
    }

    /**
     * Convert an English sentence into a queue of ISL video animations
     */
    playSign(text) {
        if (!this.isInitialized) {
            console.warn('ISL Video Avatar is not initialized yet');
            return;
        }

        if (!text || text.trim() === '') return;

        // Always stop existing video and clear queue
        this.videoQueue = [];
        this.videoPlayer.pause();
        this.videoPlayer.currentTime = 0;

        // Parse sentence into uppercase words array
        const words = text.trim().toUpperCase().split(/\s+/);

        words.forEach(word => {
            // Remove punctuation from the word
            const cleanWord = word.replace(/[.,!?]/g, '');
            if (cleanWord === '') return;

            // 1. Check if we have a full word video
            if (this.availableVideos.has(cleanWord)) {
                this.videoQueue.push(cleanWord);
            }
            // 2. Fallback to spelling it out Letter by Letter
            else {
                for (let i = 0; i < cleanWord.length; i++) {
                    const letter = cleanWord[i];
                    if (this.availableVideos.has(letter)) {
                        this.videoQueue.push(letter);
                    }
                }
            }
        });

        // Start playback if queue isn't empty
        if (this.videoQueue.length > 0) {
            this.isPlaying = true;
            this.placeholder.style.display = 'none';
            this.videoPlayer.style.display = 'block';
            this.showSignInfo(text, `Playing sequence: ${this.videoQueue.join(' → ')}`);
            this.playNextVideo();
        } else {
            this.showSignInfo(text, 'No matching signs available');
            this.videoPlayer.style.display = 'none';
            this.placeholder.style.display = 'flex';
        }
    }

    playNextVideo() {
        if (this.videoQueue.length === 0) {
            this.isPlaying = false;
            // Optionally hide video and show placeholder when done
            // this.videoPlayer.style.display = 'none';
            // this.placeholder.style.display = 'flex';
            return;
        }

        this.isPlaying = true;
        const nextSign = this.videoQueue.shift();

        // Construct the API route to serve the video securely
        const videoSrc = `/api/video/${encodeURIComponent(nextSign)}.mp4`;

        this.videoPlayer.src = videoSrc;
        this.videoPlayer.play().catch(e => {
            console.error(`Failed to play sign ${nextSign}:`, e);
            // Skip to next if an error occurs
            this.playNextVideo();
        });
    }

    showSignInfo(name, description) {
        // Keep compatibility with app.html layout
        const infoPanel = document.getElementById('avatarSignInfo');
        if (infoPanel) {
            infoPanel.innerHTML =
                '<div style="' +
                'background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));' +
                'border: 1px solid rgba(99, 102, 241, 0.3);' +
                'border-radius: 12px;' +
                'padding: 1rem;' +
                'margin-top: 1rem;' +
                'animation: fadeIn 0.3s ease;' +
                '">' +
                '<div style="font-weight: 700; color: #6366f1; font-size: 1.1rem; margin-bottom: 0.5rem;">' +
                name +
                '</div>' +
                '<div style="font-size: 0.875rem; color: #94a3b8; line-height: 1.5;">' +
                description +
                '</div>' +
                '</div>';
        }
    }

    showFallback(msg) {
        if (!this.placeholder) return;
        this.placeholder.innerHTML =
            '<i class="fa-solid fa-triangle-exclamation" style="font-size: 3rem; margin-bottom: 1rem; color: #ef4444;"></i>' +
            '<div style="font-size: 1rem; margin-bottom: 0.5rem;">System Error</div>' +
            '<div style="font-size: 0.875rem; opacity: 0.7;">' + msg + '</div>';
    }
}

// Preserve instantiation hook for app.html's initAvatar()
window.ISLAvatar = class extends ISLVideoAvatar {
    constructor(containerId) {
        super(containerId, 'islVideoPlayer', 'videoPlaceholder');
    }
};
